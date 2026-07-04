# Decision-Aware Suffix Prediction and Reasoning of Business Processes

## Abstract
Suffix prediction forecasts the remaining sequence of events of a running case until completion. Most approaches rely on deep neural networks trained on event logs, which, on average, predict well but struggle when given short prefixes or when the target suffix belongs to a rare process variant. In such scenarios, the correct path may cross multiple branching decisions, determined primarily by case- and event-level attributes, a signal that DNN-based suffix prediction models tend to underweight. Decision mining extracts rules for such branching decisions from the event log, but has so far been applied only to post-hoc and what-if analysis, not to suffix prediction. We therefore extend suffix prediction with decision mining, introducing a decision-aware suffix prediction framework as a neuro-symbolic approach that additionally enables reasoning about predicted suffixes using mined decision rules. Experiments on three of four event logs with three different suffix predictors show that the framework can improve suffix prediction performance, especially for short prefixes, and adds intrinsic interpretability.

## Repository Summary

This repository implements decision-aware suffix prediction for business processes. Given a running case
(a prefix of events), the models predict the remaining suffix. Decisions discovered from a data-aware Petri
net are injected in two complementary ways: as a **semantic loss during training** and as **decision-rule-guided
reasoning during decoding**. The whole workflow, data preparation, decision mining, model training and
evaluation, is driven from a single per-dataset notebook and a shared `experiments` package.

## Repository layout

```
src/
  data_processing/           # event-log encoding, prefix building, decision labeling, Petri-net replay
  decision_mining/           # alignment-based decision discovery + CatBoost guard estimators
  simulator/                 # generator for the synthetic Procurement event log
  suffix_pred/
    models/                  # FS_LSTM, GAN_LSTM, K_UED_LSTM architectures
    train.py                 # trainers (CTraining, TTraining, UEDTrainer)
    inference.py             # decoders (mode / probabilistic MCSA / beam)
    decision_rule_guided_reasoning_inference.py  # guided decoding
    evalaution/              # suffix decoding over the test set + evaluation metrics
    experiments/             # orchestration: configs, data_loading, decision_mining, training, evaluation
  notebooks/
    pipeline_<Dataset>.ipynb # one end-to-end pipeline per dataset
    run_all_pipelines.ipynb  # runs every pipeline notebook sequentially
```

All output directories `data/`, `models/` and `eval_results/` are created automatically by the pipeline.

## Setting up the Python environment with Pipenv

This project uses `pipenv` for dependency management. You need **Python 3.12** and **Pipenv** installed
(the `Pipfile`/`Pipfile.lock` pin Python 3.12).

Run from the **project root** (where the `Pipfile` lives):

```bash
pipenv install     # create the virtual environment and install dependencies
pipenv shell       # activate it
```

## Event logs

The real-world event logs are **not** in the repo. They are available from the original sources (see the references in the paper) and must be add to the repo. Check in the jupyters thhe pathd and change the `raw_root` in `experiments/configs.py` to point to the location.


| Dataset       | Expected raw file                        |
| ------------- | ---------------------------------------- |
| `Helpdesk`    | `helpdesk.csv`                           |
| `Sepsis`      | `Sepsis.csv`                             |
| `Procurement` | `procurement_event_log.csv`              |
| `BPIC20 DD`   | `DomesticDeclarations.csv`               |

The **Procurement** log is synthetic and can be regenerated with `src/simulator/artificial_procurement.py` (a generated copy is checked in at `src/simulator/procurement_event_log.csv`).

## Running the framework

Each dataset has a self-contained pipeline notebook in `src/notebooks/`
(`pipeline_Helpdesk.ipynb`, `pipeline_Sepsis.ipynb`, `pipeline_Procurement.ipynb`, `pipeline_BPIC20_DD.ipynb`).
Run the notebooks **from the `src/notebooks/` directory** — the first cell adds `../` to `sys.path` so that
`import suffix_pred.experiments` resolves.

Every pipeline runs five stages, each toggled by a `RUN_*` switch in the first code cell:

1. **`RUN_BASE`** — encode the raw log into "normal" tensor datasets and discover the Petri net.
2. **`RUN_MINING`** — alignment-based decision discovery + per-place guard estimators.
3. **`RUN_LABELING`** — build the decision-labeled tensor datasets (needs stage 2 output).
4. **`RUN_TRAINING`** — train the checkpoints (slow; overwrites `models/`).
5. **`RUN_EVAL`** — decode the test set and compute metrics (slow; overwrites the eval cache).

The trained architectures (`MODELS`) and evaluation conditions (`Variant`) are:

- **Models:** `UED` (Dropout-Uncertainty Encoder-Decoder LSTM), `FS` (Full-Shared next-event LSTM),
  `GAN` (Taymouri adversarial LSTM).
- **Variants:** `clean` (baseline), `decision_train` (semantic-loss training), `decision_decoding`
  (decision-rule-guided decoding of the clean model), `decision_train_decode` (both).

### Run every dataset at once

`src/notebooks/run_all_pipelines.ipynb` executes each pipeline notebook in an isolated subprocess.

## Configuration

`src/suffix_pred/experiments/configs.py` is the single source of truth for every per-(dataset, model, variant)
difference: attribute lists, concept names, hyperparameters and all path conventions. Paths for tensors,
checkpoints and eval caches are derived there by convention, so adding a dataset or model means editing that
file only.
