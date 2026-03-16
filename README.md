# Decision-Aware Suffix Prediction of Business Processes with Decision-Rule-Guided Reasoning

## Setting Up the Python Environment with Pipenv

This project uses `pipenv` for managing Python dependencies. Follow the steps below to set up the virtual environment and install the necessary packages using the provided `Pipfile`.

### Prerequisites
Make sure you have Python and Pipenv installed.

### Setup Instructions

1. **Create the Virtual Environment**:
    
    ```bash
    pipenv install
    ```

2. **Activate the Virtual Environment**:
    
    ```bash
    pipenv shell
    ```

3. **Run the Project**: Inside the virtual environment, you have the Python packages installed for running the code.


## Run the Decision-Aware Suffix Prediction of Business Processes with Decision-Rule-Guided Reasoning Framework: Data, Train and Evaluate.

### Required Directory Structure per Dataset

Before running any pipeline steps, the following directories must exist for each dataset (e.g. `<Dataset>` = `Helpdesk`, `Sepsis`, `Procurement`, `BPIC20_DD`):

```
data/<Dataset>/
    Petri_net/
        data_aware_Petri_net/
            models/
    raw_data/
    tensor_data/
        decision_labeled/
        normal/

models/<Dataset>/
    clean/
    decision/

eval_results/<Dataset>/
    clean/
    decision_decoding/
    decision_train/
```

Create all directories for a new dataset with:

```bash
DATASET=<Dataset>
mkdir -p \
  data/$DATASET/Petri_net/data_aware_Petri_net/models \
  data/$DATASET/raw_data \
  data/$DATASET/tensor_data/decision_labeled \
  data/$DATASET/tensor_data/normal \
  models/$DATASET/clean \
  models/$DATASET/decision \
  eval_results/$DATASET/clean \
  eval_results/$DATASET/decision_decoding \
  eval_results/$DATASET/decision_train
```

### Pipeline Execution Order

The notebooks must be executed in the following order for each dataset:

1. **Base Loader** (`src/notebooks/suffix_prediction/data_loader/<Dataset>_base_loader.ipynb`)  
   Preprocesses the raw event log, creates prefix dataframes, discovers the Petri net, and saves the "normal" (non-decision-labeled) tensor datasets.

2. **Decision Mining** (`src/notebooks/decision_mining/<Dataset>_decision_mining.ipynb`)  
   Runs alignment-based decision discovery on the Petri net, trains decision models per decision place, extracts probabilistic guards, and saves the model artifacts.

3. **Decision Labeling Loader** (`src/notebooks/suffix_prediction/data_loader/<Dataset>_decision_labeling_loader.ipynb`)  
   Loads the normal tensor datasets and decision mining artifacts, computes decision labels for each prefix, and saves the decision-labeled tensor datasets.

4. **Training** (`src/notebooks/suffix_prediction/trainer/<Dataset>/`)  
   Trains the suffix prediction models (C-LSTM, T-GAN-LSTM, K-UED-LSTM) in both the `clean/` (baseline) and `decision/` (decision-aware) variants.

5. **Evaluation** (`src/notebooks/suffix_prediction/evaluation/<Dataset>/`)  
   Evaluates the trained models under `clean/`, `decision_decoding/`, and `decision_train/` conditions and writes results to `eval_results/<Dataset>/`.