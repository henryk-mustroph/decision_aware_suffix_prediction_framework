import os
import pickle
import random
import concurrent.futures
from dataclasses import dataclass
from typing import Literal, Optional

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

from tqdm.auto import tqdm

from ..inference import get_evaluator

DecodeMode = Literal["mean", "mode", "probabilistic", "beam"]


@dataclass
class DecodingConfig:
    concept_name: str = "Activity"
    eos_value: str = "EOS"
    probabilistic_samples: Optional[int] = 100
    beam_width: Optional[int] = 3
    num_processes: Optional[int] = 8


# ---------------------------------------------------------------------------
# Parallel probabilistic worker helpers (module-level for pickling)
# ---------------------------------------------------------------------------

_PROB_WORKER_DECODER = None


def _init_probabilistic_worker(
    model,
    dataset,
    concept_name: str,
    eos_value: str,
    samples_per_case: int,
):
    """Create one probabilistic decoder per process to avoid per-task reinitialization."""
    global _PROB_WORKER_DECODER
    _PROB_WORKER_DECODER = get_evaluator(
        kind="mcsa",
        model=model,
        dataset=dataset,
        concept_name=concept_name,
        eos_value=eos_value,
        samples_per_case=samples_per_case,
        sample_argmax=False,
        use_variance_cat=True,
        variational_dropout_sampling=False,
    )


def _collect_probabilistic_case_chunk(case_ids: list[str]) -> list[dict]:
    """Collect probabilistic suffix samples for a chunk of case ids in a worker."""
    if _PROB_WORKER_DECODER is None:
        raise RuntimeError("Probabilistic worker decoder is not initialized.")

    rows: list[dict] = []
    decoder = _PROB_WORKER_DECODER

    for case_id in case_ids:
        full_case = decoder.cases.get(case_id)
        if full_case is None:
            continue

        for prefix_len, prefix, zero_mask, statics, suffix in decoder._iterate_case(full_case):
            prefix_activity = decoder._decode_activity_prefix(prefix)
            target_suffix = decoder._decode_activity_suffix(suffix)
            decoded_suffixes = decoder.predict_probabilistic_suffix(
                prefix=prefix,
                prefix_len=prefix_len,
                static_inputs=statics,
                mask=zero_mask,
                include_model_states=False,
            )

            rows.append(
                {
                    "case_id": case_id,
                    "prefix_len": int(prefix_len),
                    "prefix": prefix_activity,
                    "target_suffix": target_suffix,
                    "decoded_suffixes": decoded_suffixes,
                    "mode": "probabilistic",
                }
            )

    return rows


# ---------------------------------------------------------------------------
# Main decoder class
# ---------------------------------------------------------------------------


class TestSetSuffixDecoder:
    """Decode / sample suffixes for every prefix in a test dataset.

    Supports modes: ``"mean"``, ``"mode"``, ``"probabilistic"``, ``"beam"``.
    Results are returned as a list of dicts and can optionally be persisted
    to a pickle file.
    """

    def __init__(self, model, dataset, config: DecodingConfig | None = None):
        self.model = model
        self.dataset = dataset
        self.config = config or DecodingConfig()

    # -- decoder factory -----------------------------------------------------

    def _build_decoder(self, mode: DecodeMode):
        common = {
            "model": self.model,
            "dataset": self.dataset,
            "concept_name": self.config.concept_name,
            "eos_value": self.config.eos_value,
        }

        if mode == "mode":
            return get_evaluator(kind="mode", **common)

        if mode == "mean":
            return get_evaluator(
                kind="mcsa",
                samples_per_case=1,
                sample_argmax=True,
                use_variance_cat=False,
                variational_dropout_sampling=False,
                **common,
            )

        if mode == "probabilistic":
            return get_evaluator(
                kind="mcsa",
                samples_per_case=self.config.probabilistic_samples,
                sample_argmax=False,
                use_variance_cat=True,
                variational_dropout_sampling=False,
                **common,
            )

        if mode == "beam":
            return get_evaluator(
                kind="beam",
                beam_width=self.config.beam_width,
                **common,
            )

        raise ValueError(
            "Unsupported mode. Use one of: 'mean', 'mode', 'probabilistic', 'beam'."
        )

    # -- public API ----------------------------------------------------------

    def decode(
        self,
        mode: DecodeMode,
        random_order: bool = False,
        cache_path: Optional[str] = None,
        reuse_cache: bool = False,
        parallel_inference: bool = True,
        num_processes: Optional[int] = None,
    ) -> list[dict]:
        """Decode suffixes for every prefix in the test set.

        Parameters
        ----------
        mode : DecodeMode
            One of ``"mean"``, ``"mode"``, ``"probabilistic"``, ``"beam"``.
        random_order : bool
            Shuffle the order of cases before decoding.
        cache_path : str, optional
            Path to a ``.pkl`` file.  When provided the results are persisted
            after inference completes.
        reuse_cache : bool
            When *True* and *cache_path* points to an existing file, the
            cached results are returned without re-running inference.
        parallel_inference : bool
            Enable multi-process inference for the ``"probabilistic"`` mode.
        num_processes : int, optional
            Number of worker processes (probabilistic mode only).

        Returns
        -------
        list[dict]
            Each dict contains:
            ``case_id``, ``prefix_len``, ``prefix``, ``target_suffix``,
            ``decoded_suffixes``, ``mode``.
        """
        if cache_path is not None and reuse_cache:
            try:
                with open(cache_path, "rb") as handle:
                    cached = pickle.load(handle)
                if isinstance(cached, list):
                    return cached
            except FileNotFoundError:
                pass

        if mode == "probabilistic":
            outputs = self._decode_probabilistic(
                random_order=random_order,
                parallel_inference=parallel_inference,
                num_processes=num_processes,
            )
        else:
            outputs = self._decode_deterministic(mode=mode, random_order=random_order)

        if cache_path is not None:
            with open(cache_path, "wb") as handle:
                pickle.dump(outputs, handle)

        return outputs

    # -- internal helpers ----------------------------------------------------

    def _decode_deterministic(
        self, mode: DecodeMode, random_order: bool = False
    ) -> list[dict]:
        """Decode suffixes using a deterministic (non-probabilistic) decoder."""
        decoder = self._build_decoder(mode)
        outputs: list[dict] = []
        for case_id, prefix_len, prefix, target_suffix, decoded_suffixes in decoder.evaluate(
            random_order=random_order,
        ):
            outputs.append(
                {
                    "case_id": case_id,
                    "prefix_len": int(prefix_len),
                    "prefix": prefix,
                    "target_suffix": target_suffix,
                    "decoded_suffixes": decoded_suffixes,
                    "mode": mode,
                }
            )
        return outputs

    def _decode_probabilistic(
        self,
        random_order: bool = False,
        parallel_inference: bool = True,
        num_processes: Optional[int] = None,
    ) -> list[dict]:
        """Decode suffixes using probabilistic Monte-Carlo sampling."""
        decoder = self._build_decoder("probabilistic")
        case_ids = list(decoder.cases.keys())
        if random_order:
            random.shuffle(case_ids)

        worker_count = max(1, int(num_processes or self.config.num_processes or 1))
        use_parallel = parallel_inference and worker_count > 1 and len(case_ids) > 1

        samples: list[dict] = []

        if not use_parallel:
            for case_id, prefix_len, prefix, target_suffix, decoded_suffixes in decoder.evaluate(
                random_order=random_order,
            ):
                samples.append(
                    {
                        "case_id": case_id,
                        "prefix_len": int(prefix_len),
                        "prefix": prefix,
                        "target_suffix": target_suffix,
                        "decoded_suffixes": decoded_suffixes,
                        "mode": "probabilistic",
                    }
                )
        else:
            chunk_size = max(1, (len(case_ids) + worker_count - 1) // worker_count)
            case_chunks = [
                case_ids[i : i + chunk_size]
                for i in range(0, len(case_ids), chunk_size)
            ]

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_probabilistic_worker,
                initargs=(
                    self.model,
                    self.dataset,
                    self.config.concept_name,
                    self.config.eos_value,
                    int(self.config.probabilistic_samples or 0),
                ),
            ) as executor:
                futures = [
                    executor.submit(_collect_probabilistic_case_chunk, case_chunk)
                    for case_chunk in case_chunks
                ]
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Probabilistic inference chunks",
                ):
                    chunk_rows = future.result()
                    samples.extend(chunk_rows)

        return samples
