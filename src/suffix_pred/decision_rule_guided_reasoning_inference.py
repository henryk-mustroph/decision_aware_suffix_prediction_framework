from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# performance imports for torch: torch kernel uses one core only.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1" 
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm

# required files:
# use the decision labeling same as for training but online.
from data_processing.decision_labeling import DecisionLabeler
# use the inference modes implemented also in the standard case
from .inference import Mode, Beam, MCSA

_LAG_SUFFIX_RE = re.compile(r"(_past_avg|_past_mode)$")
_NUMERIC_COND_RE = re.compile(r"\(([\w]+)\s*(<=|>)\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\)")

def _strip_lag_suffix(feature_full_name: str) -> str:
    return _LAG_SUFFIX_RE.sub("", str(feature_full_name))


def _is_time_feature(base_feature_name: str) -> bool:
    return "time" in str(base_feature_name).lower()


def inverse_transform_for_display(feature_full_name: str,
                                   scaled_value: Any,
                                   numeric_scalers: Dict[str, Any]) -> Optional[float]:
    """
    Inverse-transform a scaled scalar back to its original-units value using
    the scaler registered for the base feature. Returns the input unchanged
    when no scaler applies. Non-numeric input -> None.
    """
    if scaled_value is None:
        return None
    try:
        scaled = float(scaled_value)
    except (TypeError, ValueError):
        return None
    if not numeric_scalers:
        return scaled
    base = _strip_lag_suffix(feature_full_name)
    scaler = numeric_scalers.get(base)
    if scaler is None or not hasattr(scaler, "inverse_transform"):
        return scaled
    try:
        return float(np.asarray(scaler.inverse_transform([[scaled]])).reshape(-1)[0])
    except Exception:
        return scaled


def format_duration_seconds(seconds: float) -> str:
    """
    Render a seconds value as a human-readable duration. 
    Small negatives, which arise from float arithmetic on the mean of z-scaled times (first events of a case have raw=0, z ~= -1.25, and averaging a handful of such values can drift fractionally below the scaler's zero), are clamped to 0 since negative elapsed times are physically impossible.
    """
    s = float(seconds)
    # Clamp tiny negatives that arise from float rounding of z-score
    # arithmetic. Anything more negative than 1 second is genuinely odd and
    # we surface it with a leading minus sign so the user notices.
    if -1.0 < s < 0.0:
        s = 0.0
    sign = "-" if s < 0 else ""
    s = abs(s)
    if s >= 86400.0:
        return f"{sign}{s / 86400.0:.2f} days"
    if s >= 3600.0:
        return f"{sign}{s / 3600.0:.2f} hours"
    if s >= 60.0:
        return f"{sign}{s / 60.0:.2f} minutes"
    return f"{sign}{s:.2f} seconds"


def format_number_for_display(value: float) -> str:
    """
    Render a non-time continuous attribute (e.g. BPIC2020 ``case:Amount``) in a
    human-readable way, the generic counterpart to :func:`format_duration_seconds`
    for durations. The goal is the same: surface the value in its original units
    so the reasoning trace is interpretable, rather than the raw ``%g`` output
    which collapses larger magnitudes into scientific notation (``1.234e+04``).

    Behaviour:
      - integral values  -> thousands-grouped integer (``12,340``)
      - fractional values -> thousands-grouped, two decimals (``12,340.50``)
      - extreme magnitudes (>= 1e9 or tiny non-zero < 1e-4) fall back to compact
        scientific, where grouping would be unreadable anyway.
    """
    v = float(value)
    a = abs(v)
    if a != 0.0 and (a >= 1e9 or a < 1e-4):
        return f"{v:.4g}"
    if v.is_integer():
        return f"{int(v):,d}"
    return f"{v:,.2f}"


def format_value_for_display(feature_full_name: str,
                              value: Any,
                              numeric_scalers: Dict[str, Any]) -> str:
    """
    Render a feature value for a human-readable reasoning trace:
    - categorical (str) values pass through unchanged
    - numeric values are inverse-transformed via the registered scaler
    - columns whose base name contains 'time' are rendered as durations
    - any other continuous attribute is rendered with thousands grouping and original-unit precision (see :func:`format_number_for_display`)
    """
    if value is None:
        return "None"
    if isinstance(value, str):
        return value
    raw = inverse_transform_for_display(feature_full_name, value, numeric_scalers)
    if raw is None:
        return str(value)
    base = _strip_lag_suffix(feature_full_name)
    if _is_time_feature(base):
        return format_duration_seconds(raw)
    return format_number_for_display(raw)


def render_rule_for_display(rule_str: str,
                             numeric_scalers: Dict[str, Any]) -> str:
    """
    Re-render a rule string so numeric thresholds are shown in original units (and as durations for *_time features). Categorical 'X in {...}' conditions pass through unchanged.
    """
    if not rule_str:
        return rule_str

    def _sub(m: "re.Match[str]") -> str:
        feat, op, num_s = m.group(1), m.group(2), m.group(3)
        try:
            scaled = float(num_s)
        except ValueError:
            return m.group(0)
        return f"({feat} {op} {format_value_for_display(feat, scaled, numeric_scalers)})"

    return _NUMERIC_COND_RE.sub(_sub, rule_str)


@dataclass
class DecisionGuidanceConfig:
    """
    Parameters for local decision-guided reweighting at inference.
    """
    epsilon: float = 1e-3
    beta_max: float = 2.0
    alpha: float = 0.1
    support_threshold: float = 0.05

    # Reweight a decision step ONLY when guidance is trustworthy. All three
    # gates default to no-ops so existing behaviour (always guide) is preserved
    # unless explicitly configured. See `_masked_distribution`.
    #
    # (a) Base-model uncertainty: skip guidance when the model's own next-event
    #     distribution is already peaked (normalised entropy < this). The model
    #     is confident, so steering can only perturb a good prediction (this is
    #     the BPIC20-style regression). 0.0 => never skip on this ground.
    min_base_entropy: float = 0.0
    # (b) Decision-model confidence: skip guidance when the mined model's top
    #     branch probability c_i = max_a z_i(a) < this, i.e. it has nothing
    #     decisive to say. 0.0 => never skip on this ground.
    min_decision_confidence: float = 0.0
    # (c) Observability: only guide while the decode step index <= this. At
    #     step 0 the deciding event/attributes come from the OBSERVED prefix;
    #     from step 1 on they are autoregressively PREDICTED (lab values, times,
    #     ...), so z_i is built on guesses and steering compounds the error (the
    #     Sepsis collapse). None => no cutoff (guide at every step).
    max_guided_steps: Optional[int] = None

class DecisionRuleGuidedMixin:
    """
    Shared decision-guidance and reasoning helpers for all decoders.
    """
    def _init_decision_guidance(self,
                                decision_labeler: DecisionLabeler,
                                guidance_config: Optional[DecisionGuidanceConfig] = None,
                                decision_places_bundle_path: Optional[str] = None) -> None:
        
        self.decision_labeler = decision_labeler
        self.guidance_config = guidance_config or DecisionGuidanceConfig()
        self._reasoning_bundle_guards = self._load_reasoning_guards(decision_places_bundle_path)

        self._activity_name_to_id = dict(self.dataset.all_categories[0][self.concept_name_id][2])

        self._cat_feature_names = [str(cat[0]) for cat in self.dataset.all_categories[0]]
        self._num_feature_names = [str(num[0]) for num in self.dataset.all_categories[1]]

        self._static_cat_feature_names: List[str] = []
        self._static_num_feature_names: List[str] = []
        if hasattr(self.dataset, "all_static_categories") and self.dataset.all_static_categories is not None:
            if len(self.dataset.all_static_categories) >= 1:
                self._static_cat_feature_names = [str(cat[0]) for cat in self.dataset.all_static_categories[0]]
            if len(self.dataset.all_static_categories) >= 2:
                self._static_num_feature_names = [str(num[0]) for num in self.dataset.all_static_categories[1]]

        self.last_reasoning_trace: List[Dict[str, Any]] = []
        self.last_conflicts: int = 0
        self.last_decision_steps: int = 0
        self.last_explained_steps: int = 0
        # Non-trivial = decision step where at least 2 outcomes pass the support
        # threshold. Quasi-deterministic XOR places (one branch >= 1 - tau) are
        # excluded from the non-trivial explained rate because there is no real
        # data dependency to explain.
        self.last_non_trivial_decision_steps: int = 0
        self.last_non_trivial_explained_steps: int = 0
        # Explainable = non-conflicting decision step whose chosen branch has a
        # data-aware rule available (>= 1 mined guard carrying attribute
        # conditions). The rule_explained_rate (explained / explainable) is the
        # primary explainability indicator: it isolates "could we apply the rule
        # that exists?" from rule-base coverage gaps (branches with no rule) and
        # from conflicts (which are never explainable by construction).
        self.last_explainable_decision_steps: int = 0

    @staticmethod
    def _load_reasoning_guards(decision_places_bundle_path: Optional[str]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        if not decision_places_bundle_path:
            return {}
        with open(decision_places_bundle_path, "r", encoding="utf-8") as handle:
            bundle = json.load(handle)
        out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for entry in bundle:
            place_name = str(entry.get("place_name", ""))
            guards = entry.get("guards", {})
            if place_name:
                out[place_name] = guards
        return out

    def _decode_event_attrs_from_prefix_last(self, prefix: Tuple[List[torch.Tensor], List[torch.Tensor]]) -> Dict[str, Any]:
        cats, nums = prefix
        attrs: Dict[str, Any] = {}

        for i, tensor in enumerate(cats):
            raw_id = int(tensor[0, -1].item())
            if raw_id == 0:
                continue
            feature_name = self._cat_feature_names[i]
            id_to_label = {v: k for k, v in self.dataset.all_categories[0][i][2].items()}
            attrs[feature_name] = str(id_to_label.get(raw_id, raw_id))

        for i, tensor in enumerate(nums):
            feature_name = self._num_feature_names[i]
            attrs[feature_name] = float(tensor[0, -1].item())

        return attrs

    def _decode_event_attrs_from_predicted_ids(self,
                                               predicted_cat_ids: Dict[str, int],
                                               predicted_num_values: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}

        for feature_name, token_id in predicted_cat_ids.items():
            cat_idx = None
            for i, name in enumerate(self._cat_feature_names):
                if name == feature_name:
                    cat_idx = i
                    break
            if cat_idx is None or token_id == 0:
                continue
            id_to_label = {v: k for k, v in self.dataset.all_categories[0][cat_idx][2].items()}
            attrs[feature_name] = str(id_to_label.get(token_id, token_id))

        if predicted_num_values is not None:
            for feature_name, val in predicted_num_values.items():
                attrs[feature_name] = float(val)

        return attrs

    def _decode_event_attrs_from_predicted_step(self,
                                                activity_id: int,
                                                predicted_cat_ids: Optional[Dict[str, int]],
                                                predicted_num_values: Optional[Dict[str, float]],
                                                prefix: Tuple[List[torch.Tensor], List[torch.Tensor]],
                                                static_inputs: Any) -> Dict[str, Any]:
        """
        Build event attributes for the decision context using only
        model-predicted values (and carry-forward from the last prefix event
        where no prediction is available). No ground-truth suffix values are
        consulted, so the decoder never leaks future event attributes.
        """
        predicted_cat_ids = dict(predicted_cat_ids or {})
        predicted_num_values = dict(predicted_num_values or {})

        cat_ids: Dict[str, int] = {}
        num_vals: Dict[str, float] = {}

        prefix_cats, prefix_nums = prefix

        for i, feature_name in enumerate(self._cat_feature_names):
            if i == self.concept_name_id:
                cat_ids[feature_name] = int(activity_id)
                continue
            if feature_name in predicted_cat_ids:
                cat_ids[feature_name] = int(predicted_cat_ids[feature_name])
            elif i < len(prefix_cats):
                cat_ids[feature_name] = int(prefix_cats[i][0, -1].item())

        for i, feature_name in enumerate(self._num_feature_names):
            if feature_name in predicted_num_values:
                new_val = float(predicted_num_values[feature_name])
                if i < len(prefix_nums):
                    new_val = self._enforce_monotone(feature_name, new_val,
                                                     float(prefix_nums[i][0, -1].item()))
                num_vals[feature_name] = new_val
            elif i < len(prefix_nums):
                num_vals[feature_name] = float(prefix_nums[i][0, -1].item())

        next_event_attrs = self._decode_event_attrs_from_predicted_ids(predicted_cat_ids=cat_ids,
                                                                       predicted_num_values=num_vals)
        next_event_attrs.update(self._extract_static_attrs(static_inputs))
        return self.decision_labeler._filter_attributes(next_event_attrs)

    def _roll_prefix_with_predicted_attrs(self,
                                          prefix: Tuple[List[torch.Tensor], List[torch.Tensor]],
                                          activity_id: int,
                                          predicted_cat_ids: Optional[Dict[str, int]] = None,
                                          predicted_num_values: Optional[Dict[str, float]] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Roll prefix and append predicted activity. Non-activity attributes come from
        model predictions (where available) or carry-forward from the last prefix
        event - never from ground-truth suffix.
        """
        prefix_cats, prefix_nums = prefix
        predicted_cat_ids = dict(predicted_cat_ids or {})
        predicted_num_values = dict(predicted_num_values or {})

        new_cats: List[torch.Tensor] = []
        for i, cat in enumerate(prefix_cats):
            shifted = torch.roll(cat.clone(), shifts=-1, dims=1)
            if i == self.concept_name_id:
                shifted[:, -1] = activity_id
            else:
                feature_name = self._cat_feature_names[i] if i < len(self._cat_feature_names) else None
                if feature_name is not None and feature_name in predicted_cat_ids:
                    shifted[:, -1] = int(predicted_cat_ids[feature_name])
                else:
                    shifted[:, -1] = cat[:, -1]
            new_cats.append(shifted)

        new_nums: List[torch.Tensor] = []
        for i, num in enumerate(prefix_nums):
            shifted = torch.roll(num.clone(), shifts=-1, dims=1)
            feature_name = self._num_feature_names[i] if i < len(self._num_feature_names) else None
            if feature_name is not None and feature_name in predicted_num_values:
                new_val = self._enforce_monotone(feature_name,
                                                 float(predicted_num_values[feature_name]),
                                                 float(num[0, -1].item()))
                shifted[:, -1] = new_val
            else:
                shifted[:, -1] = num[:, -1]
            new_nums.append(shifted)

        return (new_cats, new_nums)

    @staticmethod
    def _enforce_monotone(feature_name: str, new_val: float, prev_val: float) -> float:
        """Clamp cumulative features (case_elapsed_time) to be non-decreasing."""
        if "case_elapsed" in str(feature_name).lower():
            return max(new_val, prev_val)
        return new_val

    def _predict_next_event(self, current_prefix: Tuple[List[torch.Tensor], List[torch.Tensor]]) -> Tuple[Dict[str, int], Dict[str, float]]:
        """
        Query the model for next-event non-activity attribute predictions.
        Returns (cat_ids, num_values) dicts; empty dicts if the model does not
        expose a `predict_next_event` helper (carry-forward used downstream).
        """
        if not hasattr(self.model, "predict_next_event") or not callable(self.model.predict_next_event):
            return {}, {}
        try:
            model_prefix = self._project_prefix_for_model(current_prefix)
        except AttributeError:
            model_prefix = current_prefix
        try:
            preds = self.model.predict_next_event(model_prefix)
        except Exception:
            return {}, {}
        cat_ids = {feat: int(t.item()) for feat, t in preds.get("cat_ids", {}).items()}
        num_values = {feat: float(t.item()) for feat, t in preds.get("num_values", {}).items()}
        return cat_ids, num_values

    def _extract_static_attrs(self, static_inputs: Any) -> Dict[str, Any]:
        if static_inputs is None:
            return {}

        out: Dict[str, Any] = {}
        cat_static = None
        num_static = None

        if isinstance(static_inputs, tuple) and len(static_inputs) == 2:
            cat_static, num_static = static_inputs
        else:
            return out

        if cat_static is not None and hasattr(cat_static, "numel") and cat_static.numel() > 0:
            for i, feature_name in enumerate(self._static_cat_feature_names):
                token_id = int(cat_static[0, i].item()) if cat_static.dim() > 1 else int(cat_static[i].item())
                if token_id == 0:
                    continue
                id_to_label = {v: k for k, v in self.dataset.all_static_categories[0][i][2].items()}
                out[feature_name] = str(id_to_label.get(token_id, token_id))

        if num_static is not None and hasattr(num_static, "numel") and num_static.numel() > 0:
            for i, feature_name in enumerate(self._static_num_feature_names):
                val = float(num_static[0, i].item()) if num_static.dim() > 1 else float(num_static[i].item())
                out[feature_name] = val

        return out

    def _build_initial_past_events(self,
                                   prefix: Tuple[List[torch.Tensor], List[torch.Tensor]],
                                   static_inputs: Any) -> List[Dict[str, Any]]:
        cats, _ = prefix
        activity_tensor = cats[self.concept_name_id][0]
        static_attrs = self._extract_static_attrs(static_inputs)

        events: List[Dict[str, Any]] = []
        for t in range(activity_tensor.shape[0]):
            token = int(activity_tensor[t].item())
            if token == 0:
                continue
            attrs = self._decode_event_attrs_from_prefix_last(([cat[:, : t + 1] for cat in prefix[0]], [num[:, : t + 1] for num in prefix[1]]))
            
            attrs.update(static_attrs)
            attrs = self.decision_labeler._filter_attributes(attrs)
            events.append(attrs)

        return events

    def _get_decision_context(self,
                           current_input_activity: str,
                           past_events: List[Dict[str, Any]]) -> Optional[Tuple[str, Dict[str, float], float]]:
        
        transitions = self.decision_labeler.transition_by_label.get(str(current_input_activity), [])
        if len(transitions) == 0:
            return None

        trans = transitions[0]
        dps = self.decision_labeler._places_after_transition(trans)
        if len(dps) == 0:
            return None

        place_name = str(dps[0])
        z_i = self.decision_labeler._predict_shallow(place_name, past_events)
        c_i = max(z_i.values(), default=0.0)
        return place_name, z_i, float(c_i)

    def _masked_distribution(self, base_probs: torch.Tensor, z_i: Dict[str, float], step_idx: int) -> torch.Tensor:
        if len(z_i) == 0:
            return base_probs

        cfg = self.guidance_config

        # Only steer when guidance is trustworthy; otherwise leave the model's
        # own distribution untouched (the step is still scored for conflicts /
        # explainability against z_i downstream, it is just not reweighted).

        # (c) Observability: once decoding rolls past `max_guided_steps`, the
        # decision context is built from PREDICTED attributes rather than the
        # observed prefix, so z_i is unreliable -> stop guiding to avoid the
        # autoregressive error chain (Sepsis-style collapse).
        if cfg.max_guided_steps is not None and step_idx > int(cfg.max_guided_steps):
            return base_probs

        # (b) Decision-model confidence: no branch clears the floor -> the mined
        # model is not decisive here, so don't steer.
        if cfg.min_decision_confidence > 0.0:
            c_i = max(z_i.values(), default=0.0)
            if float(c_i) < float(cfg.min_decision_confidence):
                return base_probs

        # (a) Base-model uncertainty: the model's own next-event distribution is
        # already peaked (low normalised entropy) -> it is confident and
        # guidance can only perturb a good prediction, so leave it.
        if cfg.min_base_entropy > 0.0:
            num_classes = int(base_probs.shape[-1])
            if num_classes > 1:
                p = base_probs.clamp_min(1e-12)
                norm_entropy = float(-(p * p.log()).sum().item()) / math.log(num_classes)
                if norm_entropy < float(cfg.min_base_entropy):
                    return base_probs

        # Guidance sharpness beta_r = beta_max * exp(-alpha * r): max strength
        # beta_max decayed by alpha at later steps r (where the data state is
        # accumulated from predicted, less reliable attributes).
        beta_r = float(cfg.beta_max) * math.exp(-float(cfg.alpha) * float(step_idx))
        if beta_r <= 0.0:
            return base_probs

        z_vec = torch.zeros_like(base_probs)
        for label, prob in z_i.items():
            token_id = self._activity_name_to_id.get(str(label), None)
            if token_id is None:
                continue
            z_vec[token_id] = float(prob)

        mask = torch.pow(float(cfg.epsilon) + (1.0 - float(cfg.epsilon)) * z_vec, beta_r)
        weighted = base_probs * mask
        denom = weighted.sum()
        if float(denom.item()) <= 0.0:
            return base_probs
        return weighted / denom

    def _is_conflict(self, chosen_activity: str, z_i: Dict[str, float]) -> Tuple[bool, List[str]]:
        if len(z_i) == 0:
            return False, []
        tau = float(self.guidance_config.support_threshold)
        supported = [a for a, p in z_i.items() if float(p) >= tau]
        if len(supported) == 0:
            return False, supported
        return str(chosen_activity) not in set(map(str, supported)), supported

    @staticmethod
    def _coerce_float_or_none(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(f):
            return None
        return f

    def _guard_matches(self, feature_row: Dict[str, Any], guard: Dict[str, Any]) -> bool:
        intervals = guard.get("intervals", {}) or {}
        allowed = guard.get("categorical_allowed", {}) or {}
        excluded = guard.get("categorical_excluded", {}) or {}

        for feat, bounds in intervals.items():
            val = self._coerce_float_or_none(feature_row.get(feat, None))
            if val is None:
                return False
            low = self._coerce_float_or_none(bounds.get("low", None))
            high = self._coerce_float_or_none(bounds.get("high", None))
            if low is not None and not (val > low):
                return False
            if high is not None and not (val <= high):
                return False

        for feat, values in allowed.items():
            val = feature_row.get(feat, None)
            if val is None:
                return False
            if str(val) not in {str(v) for v in values}:
                return False

        for feat, values in excluded.items():
            val = feature_row.get(feat, None)
            if val is None:
                continue
            if str(val) in {str(v) for v in values}:
                return False

        return True

    @staticmethod
    def _guard_has_conditions(guard: Dict[str, Any]) -> bool:
        """
        True iff the guard carries at least one data-aware condition.

        A guard with no intervals and no categorical allow/exclude sets is a
        vacuous ``(true)`` rule that explains nothing.
        """
        return (bool(guard.get("intervals"))
                or bool(guard.get("categorical_allowed"))
                or bool(guard.get("categorical_excluded")))

    def _branch_has_conditioned_rule(self, place_name: str, next_activity_label: str) -> bool:
        """
        True iff the mined rule base provides at least one *data-aware* guard
        (carrying attribute conditions) for choosing ``next_activity_label`` at
        ``place_name``.

        Used to scope the explainability denominator: a non-conflicting step
        whose chosen branch has no conditioned rule reflects a rule-base
        coverage gap, not a failure of the reasoning component, so it is
        excluded from ``explainable_decision_steps``.
        """
        guards = self._reasoning_bundle_guards.get(place_name, {}).get(str(next_activity_label), [])
        return any(self._guard_has_conditions(g) for g in guards)

    def _best_matching_guard(self,
                             place_name: str,
                             next_activity_label: str,
                             past_events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        
        guards = (self._reasoning_bundle_guards.get(place_name, {}).get(str(next_activity_label), []))
        
        if len(guards) == 0:
            return None

        feature_row = self.decision_labeler._build_feature_row(past_events)
        matches = [g for g in guards if self._guard_matches(feature_row, g)]
        if len(matches) == 0:
            return None

        matches.sort(key=lambda g: (float(g.get("score", 0.0)), float(g.get("prob_model", 0.0)), float(g.get("support", 0.0))),
                     reverse=True)
        
        return matches[0]

    def _build_attribute_checks(self,
                                feature_row: Dict[str, Any],
                                guard: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build per-attribute membership checks used in local reasoning output.
        """
        
        checks: List[Dict[str, Any]] = []

        intervals = guard.get("intervals", {}) or {}
        allowed = guard.get("categorical_allowed", {}) or {}
        excluded = guard.get("categorical_excluded", {}) or {}

        for feat, bounds in intervals.items():
            val = self._coerce_float_or_none(feature_row.get(feat, None))
            low = self._coerce_float_or_none(bounds.get("low", None))
            high = self._coerce_float_or_none(bounds.get("high", None))

            in_rule_set = val is not None
            if in_rule_set and low is not None:
                in_rule_set = bool(val > low)
            if in_rule_set and high is not None:
                in_rule_set = bool(val <= high)

            checks.append({"attr": str(feat),
                           "value": None if val is None else float(val),
                           "in_rule_set": bool(in_rule_set),
                           "rule_type": "interval",
                           "low": None if low is None else float(low),
                           "high": None if high is None else float(high)})

        for feat, values in allowed.items():
            val = feature_row.get(feat, None)
            allowed_set = {str(v) for v in values}
            in_rule_set = val is not None and str(val) in allowed_set
            
            checks.append({"attr": str(feat),
                           "value": None if val is None else str(val),
                           "in_rule_set": bool(in_rule_set),
                           "rule_type": "categorical_allowed",
                           "allowed": sorted(list(allowed_set))})

        for feat, values in excluded.items():
            val = feature_row.get(feat, None)
            excluded_set = {str(v) for v in values}
            in_rule_set = val is None or str(val) not in excluded_set
            
            checks.append({"attr": str(feat),
                           "value": None if val is None else str(val),
                           "in_rule_set": bool(in_rule_set),
                           "rule_type": "categorical_excluded",
                           "excluded": sorted(list(excluded_set))})

        return checks

    def _append_reasoning_step(self,
                               step_idx: int,
                               input_activity: str,
                               selected_activity: str,
                               decision_context: Optional[Tuple[str, Dict[str, float], float]],
                               past_events: List[Dict[str, Any]],
                               model_prob: Optional[float] = None) -> None:
        
        if decision_context is None:
            return

        place_name, z_i, c_i = decision_context
        conflict, supported = self._is_conflict(selected_activity, z_i)
        sorted_decisions = sorted(z_i.items(), key=lambda kv: float(kv[1]), reverse=True)
        top_decision_event = str(sorted_decisions[0][0]) if len(sorted_decisions) > 0 else None
        top_decision_prob = float(sorted_decisions[0][1]) if len(sorted_decisions) > 0 else None

        is_non_trivial = len(supported) >= 2

        self.last_decision_steps += 1
        if conflict:
            self.last_conflicts += 1
        if is_non_trivial:
            self.last_non_trivial_decision_steps += 1

        matched_guard = None
        attribute_checks: List[Dict[str, Any]] = []
        explanation_status = "conflict_not_supported"
        explained = False
        branch_has_rule = False

        if not conflict:
            # Does the rule base provide a data-aware rule for this branch at all?
            branch_has_rule = self._branch_has_conditioned_rule(place_name, selected_activity)
            if branch_has_rule:
                self.last_explainable_decision_steps += 1

            matched_guard = self._best_matching_guard(place_name, selected_activity, past_events)
            matched_with_conditions = (matched_guard is not None
                                       and self._guard_has_conditions(matched_guard))
            if matched_with_conditions:
                feature_row = self.decision_labeler._build_feature_row(past_events)
                attribute_checks = self._build_attribute_checks(feature_row, matched_guard)

            # A step is data-aware-explained only when a conditioned guard
            # matched and contributed >= 1 attribute check.
            if matched_with_conditions and len(attribute_checks) > 0:
                self.last_explained_steps += 1
                if is_non_trivial:
                    self.last_non_trivial_explained_steps += 1
                explanation_status = "explained"
                explained = True
            elif branch_has_rule:
                # A data-aware rule exists for this branch but none matched the
                # current (predicted) data state - a genuine reasoning miss.
                explanation_status = "rule_unmatched"
            elif matched_guard is not None:
                # Only a vacuous (true) guard exists/matched for this branch.
                explanation_status = "matched_trivial_rule"
            else:
                # The rule base has no guard for this (place, activity) branch.
                explanation_status = "no_rule_for_branch"

        self.last_reasoning_trace.append({"step": int(step_idx),
                                          "place": place_name,
                                          "input_event": str(input_activity),
                                          "next_event": str(selected_activity),
                                          "model_prob": float(model_prob) if model_prob is not None else None,
                                          "confidence": float(c_i),
                                          "decision_top_event": top_decision_event,
                                          "decision_top_prob": top_decision_prob,
                                          "supported_set": [str(a) for a in supported],
                                          "is_non_trivial": bool(is_non_trivial),
                                          "conflict": bool(conflict),
                                          "explained": bool(explained),
                                          "branch_has_rule": bool(branch_has_rule),
                                          "explanation_status": str(explanation_status),
                                          "decision_distribution": {k: float(v) for k, v in z_i.items()},
                                          "attribute_checks": attribute_checks,
                                          "matched_rule": None if matched_guard is None else {"rule": matched_guard.get("rule", ""),
                                                                                              "raw_rule": matched_guard.get("raw_rule", ""),
                                                                                              "prob_model": float(matched_guard.get("prob_model", 0.0)),
                                                                                              "support": int(matched_guard.get("support", 0)),
                                                                                              "score": float(matched_guard.get("score", 0.0))},
                                          })

    def summarize_last_reasoning(self) -> Dict[str, Any]:
        rate = 0.0
        if self.last_decision_steps > 0:
            rate = float(self.last_conflicts) / float(self.last_decision_steps)
        explained_rate = 0.0
        if self.last_decision_steps > 0:
            explained_rate = float(self.last_explained_steps) / float(self.last_decision_steps)
        # Trivial steps (one outcome dominates above 1 - support_threshold)
        # contain no real data dependency, so the non-trivial rate is the more
        # meaningful explainability number for the paper.
        trivial_decision_steps = int(self.last_decision_steps) - int(self.last_non_trivial_decision_steps)
        non_trivial_explained_rate = 0.0
        if self.last_non_trivial_decision_steps > 0:
            non_trivial_explained_rate = (float(self.last_non_trivial_explained_steps)
                                          / float(self.last_non_trivial_decision_steps))

        # Primary explainability indicator: of non-conflicting steps whose chosen
        # branch HAS a data-aware rule, the fraction we actually explained. This
        # excludes (a) conflicts (never explainable) and (b) branches with no
        # mined rule (a coverage gap, not a reasoning failure), isolating the
        # reasoning component's ability to apply the rule that exists.
        rule_explained_rate = 0.0
        if self.last_explainable_decision_steps > 0:
            rule_explained_rate = (float(self.last_explained_steps)
                                   / float(self.last_explainable_decision_steps))

        # Per-status breakdown of why steps were NOT explained:
        #   conflict_not_supported : model chose an activity outside the support
        #   no_rule_for_branch     : the chosen branch has no mined rule at all
        #   matched_trivial_rule   : only a vacuous (true) rule applies
        #   rule_unmatched         : a data-aware rule exists but none matched
        #                            the current (predicted) data state
        def _count(status):
            return sum(1 for r in self.last_reasoning_trace
                       if r.get("explanation_status") == status)

        matched_trivial_rule = _count("matched_trivial_rule")
        rule_unmatched = _count("rule_unmatched")
        no_rule_for_branch = _count("no_rule_for_branch")
        conflict_not_supported = _count("conflict_not_supported")
        # Backward-compatible alias: the old "no_matching_rule" bucket is the
        # union of the two no-data-aware-explanation outcomes.
        no_matching_rule = rule_unmatched + no_rule_for_branch

        return {"decision_steps": int(self.last_decision_steps),
                "conflicts": int(self.last_conflicts),
                "conflict_rate": float(rate),
                "explained_steps": int(self.last_explained_steps),
                "explained_rate": float(explained_rate),
                "explainable_decision_steps": int(self.last_explainable_decision_steps),
                "rule_explained_rate": float(rule_explained_rate),
                "non_trivial_decision_steps": int(self.last_non_trivial_decision_steps),
                "non_trivial_explained_steps": int(self.last_non_trivial_explained_steps),
                "non_trivial_explained_rate": float(non_trivial_explained_rate),
                "trivial_decision_steps": int(trivial_decision_steps),
                "matched_trivial_rule": int(matched_trivial_rule),
                "rule_unmatched": int(rule_unmatched),
                "no_rule_for_branch": int(no_rule_for_branch),
                "no_matching_rule": int(no_matching_rule),
                "conflict_not_supported": int(conflict_not_supported),
                "trace": list(self.last_reasoning_trace)}

    def _reset_reasoning_state(self) -> None:
        self.last_reasoning_trace = []
        self.last_conflicts = 0
        self.last_decision_steps = 0
        self.last_explained_steps = 0
        self.last_non_trivial_decision_steps = 0
        self.last_non_trivial_explained_steps = 0
        self.last_explainable_decision_steps = 0


class GuidedMode(DecisionRuleGuidedMixin, Mode):
    """
    Decision-rule-guided arg-max decoding with step-wise reasoning traces.
    """

    def __init__(self,
                 model,
                 dataset,
                 decision_labeler: DecisionLabeler,
                 guidance_config: Optional[DecisionGuidanceConfig] = None,
                 decision_places_bundle_path: Optional[str] = None,
                 concept_name: str = "concept:name",
                 eos_value: str = "EOS"):
        
        super().__init__(model=model, dataset=dataset, concept_name=concept_name, eos_value=eos_value)
        self._init_decision_guidance(
            decision_labeler=decision_labeler,
            guidance_config=guidance_config,
            decision_places_bundle_path=decision_places_bundle_path,
        )

    def decode_suffix(self, prefix, suffix, prefix_len, static_inputs=None, return_reasoning=False):
        # `suffix` is accepted for API parity only. It is never read; doing so
        # would leak ground-truth future-event attributes into the decoder.
        max_iteration = (self.dataset.encoder_decoder.window_size - self.dataset.encoder_decoder.min_suffix_size - prefix_len)

        self._reset_reasoning_state()
        current_prefix = ([t.clone() for t in prefix[0]], [t.clone() for t in prefix[1]])
        decoded: List[str] = []
        past_events = self._build_initial_past_events(current_prefix, static_inputs)

        for step_idx in range(max_iteration + 1):
            model_prefix = self._project_prefix_for_model(current_prefix)
            logits = self.model(model_prefix).squeeze(0)
            probs = F.softmax(logits, dim=-1)

            input_activity_id = int(current_prefix[0][self.concept_name_id][0, -1].item())
            input_activity = self._activity_label(input_activity_id) if input_activity_id > 0 else ""

            decision_context = None
            masked_probs = probs
            if input_activity:
                decision_context = self._get_decision_context(input_activity, past_events)
                if decision_context is not None:
                    _, z_i, _ = decision_context
                    masked_probs = self._masked_distribution(probs, z_i, step_idx)

            activity_id = int(torch.argmax(masked_probs, dim=-1).item())
            if activity_id == self.eos_id:
                break

            selected_activity = self._activity_label(activity_id)
            decoded.append(selected_activity)

            self._append_reasoning_step(step_idx=step_idx,
                                        input_activity=input_activity,
                                        selected_activity=selected_activity,
                                        decision_context=decision_context,
                                        past_events=past_events,
                                        model_prob=float(masked_probs[activity_id].item()))

            # Predict next-event non-activity attributes from the model (no GT).
            predicted_cat_ids, predicted_num_values = self._predict_next_event(current_prefix)

            current_prefix = self._roll_prefix_with_predicted_attrs(prefix=current_prefix,
                                                                     activity_id=activity_id,
                                                                     predicted_cat_ids=predicted_cat_ids,
                                                                     predicted_num_values=predicted_num_values)

            next_event_attrs = self._decode_event_attrs_from_prefix_last(current_prefix)
            next_event_attrs.update(self._extract_static_attrs(static_inputs))
            next_event_attrs = self.decision_labeler._filter_attributes(next_event_attrs)
            past_events.append(next_event_attrs)

        if return_reasoning:
            return decoded, self.summarize_last_reasoning()
        return decoded

    def evaluate(self, random_order=False, include_model_states=False, return_reasoning=False):
        self._ensure_eval_mode()
        case_items = list(self.cases.items())
        if random_order:
            import random

            case_items = random.sample(case_items, len(case_items))

        for _, (case_name, full_case) in tqdm(enumerate(case_items), total=len(self.cases)):
            for _, (prefix_len, prefix, _, statics, suffix) in enumerate(self._iterate_case(full_case)):
                prefix_activity = self._decode_activity_prefix(prefix)
                target_suffix = self._decode_activity_suffix(suffix)

                if return_reasoning:
                    decoded, reasoning = self.decode_suffix(prefix=prefix,
                                                            suffix=suffix,
                                                            prefix_len=prefix_len,
                                                            static_inputs=statics,
                                                            return_reasoning=True)
                    
                    yield (case_name, prefix_len, prefix_activity, target_suffix, [decoded], reasoning)
                else:
                    decoded = self.decode_suffix(prefix=prefix, suffix=suffix, prefix_len=prefix_len, static_inputs=statics)
                    yield (case_name, prefix_len, prefix_activity, target_suffix, [decoded])


class GuidedMCSA(DecisionRuleGuidedMixin, MCSA):
    """
    Decision-rule-guided Monte-Carlo suffix sampling.
    """

    def __init__(self,
                 model,
                 dataset,
                 decision_labeler: DecisionLabeler,
                 guidance_config: Optional[DecisionGuidanceConfig] = None,
                 decision_places_bundle_path: Optional[str] = None,
                 concept_name: str = "concept:name",
                 eos_value: str = "EOS",
                 samples_per_case: int = 100,
                 sample_argmax: bool = False,
                 use_variance_cat: bool = True,
                 variational_dropout_sampling: bool = True):
        
        super().__init__(model=model,
                         dataset=dataset,
                         concept_name=concept_name,
                         eos_value=eos_value,
                         samples_per_case=samples_per_case,
                         sample_argmax=sample_argmax,
                         use_variance_cat=use_variance_cat,
                         variational_dropout_sampling=variational_dropout_sampling)
        
        self._init_decision_guidance(decision_labeler=decision_labeler,
                                     guidance_config=guidance_config,
                                     decision_places_bundle_path=decision_places_bundle_path)

    def sample_suffix(self,
                     prefix,
                     prefix_len,
                     static_inputs,
                     mask,
                     suffix=None,
                     include_model_states=False,
                     return_reasoning=False):
        
        # call model:
        prediction, (h, c), z = self.model.inference(prefix=prefix, static_inputs=static_inputs, mask=mask)

        max_iteration = (self.dataset.encoder_decoder.window_size - self.dataset.encoder_decoder.min_suffix_size - prefix_len)

        self._reset_reasoning_state()
        sampled_suffix: List[str] = []
        model_states = [] if include_model_states else None

        past_events = self._build_initial_past_events(prefix, static_inputs)
        current_input_activity_id = int(prefix[0][self.concept_name_id][0, -1].item())

        step_idx = 0
        while step_idx <= max_iteration:
            cat_means, cat_vars = prediction[0][0], prediction[1][0]
            activity_key = f"{self.concept_name}_mean"
            if activity_key not in cat_means:
                activity_key = [k for k in cat_means.keys() if k.endswith("_mean")][0]

            activity_logits = cat_means[activity_key]
            if self.use_variance_cat:
                var_key = f"{self.concept_name}_var"
                if var_key in cat_vars:
                    logvar = torch.clamp(cat_vars[var_key], min=-6.0, max=6.0)
                    std = torch.exp(0.5 * logvar)
                    activity_logits = torch.normal(activity_logits, std)

            base_probs = F.softmax(activity_logits, dim=-1).squeeze(0)

            input_activity = self._activity_label(current_input_activity_id) if current_input_activity_id > 0 else ""
            decision_context = None
            masked_probs = base_probs
            if input_activity:
                decision_context = self._get_decision_context(input_activity, past_events)
                if decision_context is not None:
                    _, z_i, _ = decision_context
                    masked_probs = self._masked_distribution(base_probs, z_i, step_idx)

            if self.sample_argmax:
                sampled_activity_id = int(torch.argmax(masked_probs, dim=-1).item())
            else:
                sampled_activity_id = int(torch.multinomial(masked_probs, num_samples=1, replacement=True).item())

            if sampled_activity_id == self.eos_id:
                break

            selected_activity = self._activity_label(sampled_activity_id)
            sampled_suffix.append(selected_activity)

            self._append_reasoning_step(step_idx=step_idx,
                                        input_activity=input_activity,
                                        selected_activity=selected_activity,
                                        decision_context=decision_context,
                                        past_events=past_events,
                                        model_prob=float(masked_probs[sampled_activity_id].item()))

            cat_predictions = self._sample_categorical_predictions(cat_means, cat_vars)
            cat_predictions[activity_key] = torch.tensor([[sampled_activity_id]], device=activity_logits.device, dtype=torch.long)

            num_means_dict = prediction[0][1] if len(prediction[0]) > 1 else {}
            num_vars_dict = prediction[1][1] if len(prediction[1]) > 1 else {}
            num_predictions = self._sample_numerical_predictions(num_means_dict, num_vars_dict)

            if include_model_states:
                model_states.append((h, c))

            # Build decision-context features from the model-predicted attributes
            # (and carry-forward from the last prefix event where no prediction
            # exists). NEVER read from `suffix` here - that would leak future GT.
            predicted_cat_ids: Dict[str, int] = {}
            for key, value in cat_predictions.items():
                feature_name = key[:-5] if key.endswith("_mean") else key
                predicted_cat_ids[feature_name] = int(value.item())

            predicted_num_values: Dict[str, float] = {}
            for key, value in num_predictions.items():
                feature_name = key[:-5] if key.endswith("_mean") else key
                # value is a tensor; take the first scalar.
                predicted_num_values[feature_name] = float(value.reshape(-1)[0].item())

            next_event_attrs = self._decode_event_attrs_from_predicted_step(activity_id=sampled_activity_id,
                                                                              predicted_cat_ids=predicted_cat_ids,
                                                                              predicted_num_values=predicted_num_values,
                                                                              prefix=prefix,
                                                                              static_inputs=static_inputs)
            past_events.append(next_event_attrs)

            # Feed predicted attributes back into the decoder (no GT).
            next_event = (list(cat_predictions.values()), list(num_predictions.values()))

            if self.variational_dropout_sampling:
                prediction, (h, c) = self.model.inference(last_event=next_event, hx=(h, c), z=z)
            else:
                prediction, (h, c) = self.model.inference(last_event=next_event, hx=(h, c), z=None)

            current_input_activity_id = sampled_activity_id
            step_idx += 1

        if return_reasoning:
            reasoning = self.summarize_last_reasoning()
            if include_model_states:
                return sampled_suffix, model_states, reasoning
            return sampled_suffix, reasoning

        if include_model_states:
            return sampled_suffix, model_states
        return sampled_suffix

    def predict_probabilistic_suffix(self, prefix, prefix_len, static_inputs, mask, suffix=None, include_model_states=False):
        suffixes = []
        for _ in range(self.samples_per_case):
            suffixes.append(
                self.sample_suffix(prefix=prefix,
                                   prefix_len=prefix_len,
                                   static_inputs=static_inputs,
                                   mask=mask,
                                   suffix=suffix,
                                   include_model_states=include_model_states))
            
        return suffixes

    def evaluate(self, random_order=False, include_model_states=False, return_reasoning=False):
        self._ensure_eval_mode()
        case_items = list(self.cases.items())
        if random_order:
            import random

            case_items = random.sample(case_items, len(case_items))

        for _, (case_name, full_case) in tqdm(enumerate(case_items), total=len(self.cases)):
            for _, (prefix_len, prefix, zero_mask, statics, suffix) in enumerate(self._iterate_case(full_case)):
                prefix_activity = self._decode_activity_prefix(prefix)
                target_suffix = self._decode_activity_suffix(suffix)

                if not return_reasoning:
                    sampled_suffixes = self.predict_probabilistic_suffix(prefix=prefix,
                                                                         prefix_len=prefix_len,
                                                                         static_inputs=statics,
                                                                         mask=zero_mask,
                                                                         suffix=suffix,
                                                                         include_model_states=include_model_states)
                    yield (case_name, prefix_len, prefix_activity, target_suffix, sampled_suffixes)
                    continue

                sampled_suffixes: List[List[str]] = []
                reasonings: List[Dict[str, Any]] = []
                for _ in range(self.samples_per_case):
                    sampled, reasoning = self.sample_suffix(prefix=prefix,
                                                            prefix_len=prefix_len,
                                                            static_inputs=statics,
                                                            mask=zero_mask,
                                                            suffix=suffix,
                                                            include_model_states=False,
                                                            return_reasoning=True
                                                            )
                    sampled_suffixes.append(sampled)
                    reasonings.append(reasoning)

                yield (case_name, prefix_len, prefix_activity, target_suffix, sampled_suffixes, reasonings)


class GuidedBeam(DecisionRuleGuidedMixin, Beam):
    """
    Decision-rule-guided beam search with beam-local decision contexts.
    GAN LSTM
    """
    def __init__(self,
                 model,
                 dataset,
                 decision_labeler: DecisionLabeler,
                 guidance_config: Optional[DecisionGuidanceConfig] = None,
                 decision_places_bundle_path: Optional[str] = None,
                 concept_name: str = "concept:name",
                 eos_value: str = "EOS",
                 beam_width: int = 3):
        
        super().__init__(model=model, dataset=dataset, concept_name=concept_name, eos_value=eos_value, beam_width=beam_width)
        self._init_decision_guidance(decision_labeler=decision_labeler,
                                     guidance_config=guidance_config,
                                     decision_places_bundle_path=decision_places_bundle_path)

    def _next_activity_probs_from_prefix(self, prefix) -> torch.Tensor:
        model_prefix = self._project_prefix_for_model(prefix)
        logits = self.model(model_prefix)
        if logits.dim() == 3:
            logits = logits[0, 0, :]
        elif logits.dim() == 2:
            logits = logits[0, :]
        return F.softmax(logits, dim=-1)

    def decode_suffix(self, prefix, suffix, prefix_len, static_inputs=None, return_reasoning=False):
        max_iteration = (self.dataset.encoder_decoder.window_size - self.dataset.encoder_decoder.min_suffix_size - prefix_len)

        static_attrs = self._extract_static_attrs(static_inputs)
        initial_events = self._build_initial_past_events(prefix, static_inputs)

        beams: List[Dict[str, Any]] = [{"prefix": ([t.clone() for t in prefix[0]], [t.clone() for t in prefix[1]]),
                                        "seq": [],
                                        "score": 0.0,
                                        "past_events": [dict(ev) for ev in initial_events],
                                        "reasoning": [],
                                        "conflicts": 0,
                                        "decision_steps": 0,
                                        "done": False}]

        for step_idx in range(max_iteration + 1):
            candidates: List[Dict[str, Any]] = []
            for beam in beams:
                if beam["done"]:
                    candidates.append(beam)
                    continue

                current_prefix = beam["prefix"]
                probs = self._next_activity_probs_from_prefix(current_prefix)

                input_activity_id = int(current_prefix[0][self.concept_name_id][0, -1].item())
                input_activity = self._activity_label(input_activity_id) if input_activity_id > 0 else ""

                decision_context = None
                masked_probs = probs
                if input_activity:
                    decision_context = self._get_decision_context(input_activity, beam["past_events"])
                    if decision_context is not None:
                        _, z_i, _ = decision_context
                        masked_probs = self._masked_distribution(probs, z_i, step_idx)

                topk_logp, topk_idx = torch.topk(torch.log(masked_probs + 1e-12), k=min(self.beam_width, masked_probs.shape[-1]))
                for j in range(topk_idx.shape[0]):
                    tok = int(topk_idx[j].item())
                    tok_logp = float(topk_logp[j].item())

                    new_seq = beam["seq"] + [tok]
                    done = tok == self.eos_id

                    new_prefix = beam["prefix"]
                    new_past_events = [dict(ev) for ev in beam["past_events"]]
                    new_reasoning = [dict(r) for r in beam["reasoning"]]
                    new_conflicts = int(beam["conflicts"])
                    new_decision_steps = int(beam["decision_steps"])

                    if not done:
                        selected_activity = self._activity_label(tok)
                        if decision_context is not None:
                            place_name, z_i, c_i = decision_context
                            conflict, supported = self._is_conflict(selected_activity, z_i)
                            sorted_decisions = sorted(z_i.items(), key=lambda kv: float(kv[1]), reverse=True)
                            top_decision_event = str(sorted_decisions[0][0]) if len(sorted_decisions) > 0 else None
                            top_decision_prob = float(sorted_decisions[0][1]) if len(sorted_decisions) > 0 else None
                            is_non_trivial = len(supported) >= 2
                            new_decision_steps += 1
                            if conflict:
                                new_conflicts += 1

                            guard = None
                            attribute_checks: List[Dict[str, Any]] = []
                            explanation_status = "conflict_not_supported"
                            explained = False
                            branch_has_rule = False
                            if not conflict:
                                branch_has_rule = self._branch_has_conditioned_rule(place_name, selected_activity)
                                guard = self._best_matching_guard(place_name, selected_activity, new_past_events)
                                matched_with_conditions = (guard is not None
                                                           and self._guard_has_conditions(guard))
                                if matched_with_conditions:
                                    feature_row = self.decision_labeler._build_feature_row(new_past_events)
                                    attribute_checks = self._build_attribute_checks(feature_row, guard)
                                if matched_with_conditions and len(attribute_checks) > 0:
                                    explained = True
                                    explanation_status = "explained"
                                elif branch_has_rule:
                                    explanation_status = "rule_unmatched"
                                elif guard is not None:
                                    explanation_status = "matched_trivial_rule"
                                else:
                                    explanation_status = "no_rule_for_branch"

                            new_reasoning.append({"step": int(step_idx),
                                                  "place": place_name,
                                                  "input_event": str(input_activity),
                                                  "next_event": str(selected_activity),
                                                  "model_prob": float(masked_probs[tok].item()),
                                                  "confidence": float(c_i),
                                                  "decision_top_event": top_decision_event,
                                                  "decision_top_prob": top_decision_prob,
                                                  "supported_set": [str(a) for a in supported],
                                                  "is_non_trivial": bool(is_non_trivial),
                                                  "conflict": bool(conflict),
                                                  "explained": bool(explained),
                                                  "branch_has_rule": bool(branch_has_rule),
                                                  "explanation_status": str(explanation_status),
                                                  "decision_distribution": {k: float(v) for k, v in z_i.items()},
                                                  "attribute_checks": attribute_checks,
                                                  "matched_rule": None if guard is None else {"rule": guard.get("rule", ""),
                                                                                              "raw_rule": guard.get("raw_rule", ""),
                                                                                              "prob_model": float(guard.get("prob_model", 0.0)),
                                                                                              "support": int(guard.get("support", 0)),
                                                                                              "score": float(guard.get("score", 0.0))},
                                                  })

                        # Predict next-event non-activity attributes from the model
                        # (no GT). For models without a multi-head decoder we
                        # fall back to carry-forward of the last prefix value.
                        predicted_cat_ids, predicted_num_values = self._predict_next_event(beam["prefix"])

                        new_prefix = self._roll_prefix_with_predicted_attrs(
                            prefix=beam["prefix"],
                            activity_id=tok,
                            predicted_cat_ids=predicted_cat_ids,
                            predicted_num_values=predicted_num_values,
                        )
                        next_attrs = self._decode_event_attrs_from_prefix_last(new_prefix)
                        next_attrs.update(static_attrs)
                        next_attrs = self.decision_labeler._filter_attributes(next_attrs)
                        new_past_events.append(next_attrs)

                    candidates.append({"prefix": new_prefix,
                                       "seq": new_seq,
                                       "score": float(beam["score"] + tok_logp),
                                       "past_events": new_past_events,
                                       "reasoning": new_reasoning,
                                       "conflicts": new_conflicts,
                                       "decision_steps": new_decision_steps,
                                       "done": done
                                       })

            candidates.sort(key=lambda b: b["score"], reverse=True)
            beams = candidates[: self.beam_width]
            if all(b["done"] for b in beams):
                break

        decoded_beams: List[List[str]] = []
        reasoning_beams: List[Dict[str, Any]] = []

        for b in beams:
            decoded = []
            for token in b["seq"]:
                if token == 0 or token == self.eos_id:
                    break
                decoded.append(self._activity_label(int(token)))
            decoded_beams.append(decoded)

            rate = 0.0
            if b["decision_steps"] > 0:
                rate = float(b["conflicts"]) / float(b["decision_steps"])
            explained_steps = sum(1 for r in b["reasoning"] if bool(r.get("explained", False)))
            explained_rate = 0.0
            if b["decision_steps"] > 0:
                explained_rate = float(explained_steps) / float(b["decision_steps"])
            non_trivial_decision_steps = sum(1 for r in b["reasoning"] if bool(r.get("is_non_trivial", False)))
            non_trivial_explained_steps = sum(1 for r in b["reasoning"]
                                              if bool(r.get("is_non_trivial", False))
                                              and bool(r.get("explained", False)))
            non_trivial_explained_rate = 0.0
            if non_trivial_decision_steps > 0:
                non_trivial_explained_rate = float(non_trivial_explained_steps) / float(non_trivial_decision_steps)
            trivial_decision_steps = int(b["decision_steps"]) - int(non_trivial_decision_steps)

            def _count(status):
                return sum(1 for r in b["reasoning"] if r.get("explanation_status") == status)

            matched_trivial_rule = _count("matched_trivial_rule")
            rule_unmatched = _count("rule_unmatched")
            no_rule_for_branch = _count("no_rule_for_branch")
            conflict_not_supported = _count("conflict_not_supported")
            no_matching_rule = rule_unmatched + no_rule_for_branch

            # Non-conflicting steps whose chosen branch has a data-aware rule.
            explainable_decision_steps = sum(1 for r in b["reasoning"]
                                             if not bool(r.get("conflict", False))
                                             and bool(r.get("branch_has_rule", False)))
            rule_explained_rate = 0.0
            if explainable_decision_steps > 0:
                rule_explained_rate = float(explained_steps) / float(explainable_decision_steps)

            reasoning_beams.append({"decision_steps": int(b["decision_steps"]),
                                    "conflicts": int(b["conflicts"]),
                                    "conflict_rate": float(rate),
                                    "explained_steps": int(explained_steps),
                                    "explained_rate": float(explained_rate),
                                    "explainable_decision_steps": int(explainable_decision_steps),
                                    "rule_explained_rate": float(rule_explained_rate),
                                    "non_trivial_decision_steps": int(non_trivial_decision_steps),
                                    "non_trivial_explained_steps": int(non_trivial_explained_steps),
                                    "non_trivial_explained_rate": float(non_trivial_explained_rate),
                                    "trivial_decision_steps": int(trivial_decision_steps),
                                    "matched_trivial_rule": int(matched_trivial_rule),
                                    "rule_unmatched": int(rule_unmatched),
                                    "no_rule_for_branch": int(no_rule_for_branch),
                                    "no_matching_rule": int(no_matching_rule),
                                    "conflict_not_supported": int(conflict_not_supported),
                                    "trace": b["reasoning"],
                                    "beam_logprob": float(b["score"])
                                    })

        if return_reasoning:
            return decoded_beams, reasoning_beams
        return decoded_beams

    def evaluate(self, random_order=False, include_model_states=False, return_reasoning=False):
        self._ensure_eval_mode()
        case_items = list(self.cases.items())
        if random_order:
            import random

            case_items = random.sample(case_items, len(case_items))

        for _, (case_name, full_case) in tqdm(enumerate(case_items), total=len(self.cases)):
            for _, (prefix_len, prefix, _, statics, suffix) in enumerate(self._iterate_case(full_case)):
                prefix_activity = self._decode_activity_prefix(prefix)
                target_suffix = self._decode_activity_suffix(suffix)
                if return_reasoning:
                    
                    decoded_suffixes, reasonings = self.decode_suffix(prefix=prefix,
                                                                      suffix=suffix,
                                                                      prefix_len=prefix_len,
                                                                      static_inputs=statics,
                                                                      return_reasoning=True)
                    
                    yield (case_name, prefix_len, prefix_activity, target_suffix, decoded_suffixes, reasonings)
                else:
                    decoded_suffixes = self.decode_suffix(prefix=prefix, suffix=suffix, prefix_len=prefix_len, static_inputs=statics)
                    yield (case_name, prefix_len, prefix_activity, target_suffix, decoded_suffixes)


# call the right mode method:
def get_decision_guided_evaluator(kind: str,
                                  model,
                                  dataset,
                                  decision_labeler: DecisionLabeler,
                                  guidance_config: Optional[DecisionGuidanceConfig] = None,
                                  decision_places_bundle_path: Optional[str] = None,
                                  **kwargs):
    """
    Factory for decision-rule-guided evaluators.
    Supported kinds:
    - "mcsa": guided Monte-Carlo suffix sampling
    - "mode": guided arg-max decoding
    - "beam": guided beam-search decoding
    """
    kind_normalized = kind.strip().lower()

    common_kwargs = {"model": model,
                     "dataset": dataset,
                     "decision_labeler": decision_labeler,
                     "guidance_config": guidance_config,
                     "decision_places_bundle_path": decision_places_bundle_path}
    common_kwargs.update(kwargs)

    if kind_normalized == "mcsa":
        return GuidedMCSA(**common_kwargs)
    if kind_normalized == "mode":
        return GuidedMode(**common_kwargs)
    if kind_normalized == "beam":
        return GuidedBeam(**common_kwargs)

    raise ValueError(f"Unknown guided evaluator kind '{kind}'. Supported: 'mcsa', 'mode', 'beam'.")

