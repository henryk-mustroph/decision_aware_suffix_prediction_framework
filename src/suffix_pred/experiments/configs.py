"""
Configuration registry for suffix-prediction experiments.
Observed conventions (from the existing notebooks):
  petri net    : data/{key}/Petri_net/{slug}.pkl
  
  train/val    : data/{key}/tensor_data/{normal|decision_labeled}/{slug}_all_5_{train|val}.pkl
  test         : data/{key}/tensor_data/normal/{slug}_all_5_test.pkl
  
  decision art : data/{key}/Petri_net/data_aware_Petri_net/{decision_places_bundle.json, models/, numeric_scalers.pkl}
  
  model file   : models/{key}/{clean|decision}/{key}_{model_file}_v1_{clean|DA}.pkl
  eval cache   : eval_results/{key}/{variant}/{slug}_{model_slug}_{tag}_outputs.pkl
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

# Repo root: experiments/ -> suffix_pred/ -> src/ -> <repo root>
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# to the main for data loading
# Path(__file__).resolve().parents[4]

# Variants
class Variant(str, Enum):
    """
    A variant decomposes into (model source, decode strategy):
    """
    # clean-trained model, plain decode
    CLEAN = "clean"
    # decision-aware-trained model, plain decode
    DECISION_TRAIN = "decision_train"
    # clean-trained model, decision-guided decode
    DECISION_DECODING = "decision_decoding"
    # decision-aware-trained model, decision-guided decode
    DECISION_TRAIN_DECODE = "decision_train_decode"

    @property
    def model_source(self) -> str:
        """
        Which trained checkpoint to load: 'clean' or 'decision'.
        """
        return "decision" if self in (Variant.DECISION_TRAIN,
                                      Variant.DECISION_TRAIN_DECODE) else "clean"

    @property
    def decode(self) -> str:
        """
        Decoding strategy: 'plain' or 'guided'.
        """
        return "guided" if self in (Variant.DECISION_DECODING,
                                    Variant.DECISION_TRAIN_DECODE) else "plain"

    @property
    def uses_decision_loss(self) -> bool:
        """
        Whether the training run for this variant adds the semantic loss.
        """
        return self.model_source == "decision"

    
# Ensure that model configs and and datasets configs are matching for correct training and decoding.
# All initialized alues are used for the evaluation in the paper
# Model configs (Suffix pred): Intialization of default values and type
@dataclass(frozen=True)
class EventLogSpec:
    """
    Do basic data preparation here used for suffix pred and decision mining
    """
    event_log_location: str
    
    case_name: str
    
    timestamp_name: str
    date_format: Optional[str] = None
    
    cat_dynamic: List[str] = field(default_factory=list)
    cat_static: List[str]  = field(default_factory=list)
    num_dynamic: List[str] = field(default_factory=list)
    # static continuous columns (e.g. BPIC20 case:Amount)
    num_static: List[str] = field(default_factory=list)

    # sequence length training: determine size
    min_suffix_size: int = 5

    # size of train/ val/ and test set
    train_validation_size: float = 0.15
    test_validation_size: float = 0.2
    
    # window size: prefix padding for consistent length
    window_size: object = "auto"

    # Inductive-Miner noise filter used for Petri-net discovery (IMf). 0.0 keeps
    # every observed variant (clean logs); raising it filters infrequent
    # behaviour, yielding a simpler, higher-precision net with fewer spurious
    # decision places and less silent routing. This is the per-dataset sweep
    # knob: edit it here and re-run base building. build_base_dataset reads this
    # unless an explicit net_noise_threshold argument overrides it.
    net_noise_threshold: float = 0.0

    # time values
    time_since_case_start_column: str = "case_elapsed_time"
    time_since_last_event_column: str = "event_elapsed_time"
    day_in_week_column: str = "day_in_week"
    seconds_in_day_column: str = "seconds_in_day"
    
    # 
    clean_activity_by_suffix: bool = False
    
    # uncleaned CSV when cleaning: needed for BPIC2020
    raw_source_location: Optional[str] = None    

    def to_event_log_properties(self, concept_name: str) -> Dict[str, object]:
        """
        Build the event_log_properties:
        """
        props: Dict[str, object] = {# Case id
                                    "case_name": self.case_name,
                                    # Event label name
                                    "concept_name": concept_name,
                                    # time features
                                    "timestamp_name": self.timestamp_name,
                                    "time_since_case_start_column": self.time_since_case_start_column,
                                    "time_since_last_event_column": self.time_since_last_event_column,
                                    "day_in_week_column": self.day_in_week_column,
                                    "seconds_in_day_column": self.seconds_in_day_column,
                                    # seq. length (training)
                                    "min_suffix_size": self.min_suffix_size,
                                    # dataset sizes
                                    "train_validation_size": self.train_validation_size,
                                    "test_validation_size": self.test_validation_size,
                                    # pading size
                                    "window_size": self.window_size,
                                    # structure of case and event-level attributes needed for preparation and encoding
                                    "categorical_columns": list(self.cat_dynamic),
                                    "static_categorical_columns": list(self.cat_static),
                                    "continuous_columns": list(self.num_dynamic),
                                    "static_continuous_columns": list(self.num_static)}
        
        if self.date_format is not None:
            props["date_format"] = self.date_format
        return props


@dataclass(frozen=True)
class ModelFeatures:
    """
    Explicit input/output feature selection for the NN models:
    - input_*  - model input features        
    - output_* - model output features, the model predictions.
    - static_* - case-level inputs encoded in the separate static path (UED only)
    """
    # features the model reads (dynamic); input_cat includes the activity
    input_cat: List[str] = field(default_factory=list)
    input_num: List[str] = field(default_factory=list)
    # case-level inputs (only encoded when use_statics=True)
    static_cat: List[str] = field(default_factory=list)
    static_num: List[str] = field(default_factory=list)
    # features the model predicts (dynamic); output_cat includes the activity
    output_cat: List[str] = field(default_factory=list)
    output_num: List[str] = field(default_factory=list)
    # encode the static path (UED) or not (FS / GAN)
    use_statics: bool = False


@dataclass(frozen=True)
class DatasetConfig:
    # e.g. "Helpdesk"
    key: str
    # lowercase file stem, e.g. "helpdesk"
    slug: str
    # activity column ("Activity" | "concept:name")
    concept_name: str
    # decision-model dynamic features (subset)
    dynamic_attributes: List[str]
    # decision-model static features (subset)
    static_attributes: List[str]
    # raw-log -> tensor encoding spec
    event_log: EventLogSpec
    # explicit per-architecture input/output feature lists, keyed by model
    # key ("UED" | "FS" | "GAN"). Edit these to change what each model reads/predicts.
    model_features: Dict[str, ModelFeatures] = field(default_factory=dict)
    # endo of sequence value
    eos_value: str = "EOS"

    @property
    def result_name(self) -> str:
        """
        File stem used by the loaders, e.g. 'helpdesk_all'.
        """
        return f"{self.slug}_all"

# Model configuration: important hyperparameters and design decisions for the model implementations 
@dataclass(frozen=True)
class ModelConfig:
    # "UED": Uncertainty-aware encoder decoder  | "FS": fully shared LSTM | "GAN": generative adversarial LSTM
    key: str                      
    model_file: str               
    model_slug: str               
    
    # default hyperparams:
    hidden_size: int = 128
    num_layers: int = 4
    dropout: float = 0.1
    
    # Default values:
    # Decision-aware defaults (only used when variant.uses_decision_loss).
    lambda_sem: float = 0.2
    tau: float = 0.3
    
    
    # Common optimisation knobs: Used similar across models
    learning_rate: float = 1e-5
    epochs: int = 100
    batch_size: int = 128

    # Decision-aware warm-start fine-tuning (only applied when variant.uses_decision_loss AND a clean checkpoint exists):
    # load the clean weights, then fine-tune gently so the semantic loss does not destroy the
    # learned activity distributions. 
    # 
    # This is what makes decision training use a
    # different lr/epochs than clean training.
    # warm_start : enable/disable the warm-start fine-tune.
    # warm_start_lr_factor : learning_rate is multiplied by this for the fine-tune.
    # fine_tune_epochs : epochs for the fine-tune; None => max(10, epochs // 5).
    warm_start: bool = True
    warm_start_lr_factor: float = 0.2
    fine_tune_epochs: Optional[int] = None

    # Architecture-specific knobs (optimizer kind, GAN/Gumbel/reg/TF params, ...).
    extra: Dict[str, object] = field(default_factory=dict)


# Initialization of: Prediction features
# datasets (for decision labeling and mininig), 
# event log spec (for preparation and suffix prediction) and
DATASETS: Dict[str, DatasetConfig] = {
    # Helpdesk dataset: Values also used for decision mining!
    "Helpdesk": DatasetConfig(key="Helpdesk",
                              slug="helpdesk",
                              
                              # event label
                              concept_name="Activity",
                              
                              # event-level attributes for decision mining
                              dynamic_attributes=["Resource",
                                                  "case_elapsed_time"
                                                  #"event_elapsed_time",
                                                  ],
                              
                              # case-level attributes for decision mining:                 
                              static_attributes=[# "VariantIndex",
                                                 "seriousness",
                                                 "customer",
                                                 "product",
                                                 "responsible_section",
                                                 # "seriousness_2",
                                                 # "service_level",
                                                 # "service_type",
                                                 # "support_section",
                                                 # "workgroup"
                                                 ],
                              
                              # for suffix prediction:
                              event_log=EventLogSpec(event_log_location="data/data/helpdesk.csv",
                                                     # case id
                                                     case_name="CaseID",
                                                     # time info
                                                     timestamp_name="CompleteTimestamp",
                                                     date_format="%Y/%m/%d %H:%M:%S.%f",                   
                                  
                                                     # dynamic categorical attributes
                                                     cat_dynamic=["Activity",
                                                                  "Resource"],
                                                     
                                                     # continuous dynamic attributes               
                                                     num_dynamic=["case_elapsed_time",
                                                                  # "event_elapsed_time",
                                                                  "day_in_week",
                                                                  "seconds_in_day"],
                                                     
                                                     # categorical static attributes
                                                     cat_static=["VariantIndex",
                                                                 "seriousness",
                                                                 "customer",
                                                                 "product",
                                                                 "responsible_section",
                                                                 "seriousness_2",
                                                                 "service_level",
                                                                 "service_type",
                                                                 "support_section",
                                                                 "workgroup"
                                                                 ],
                                                     
                                                     num_static = []),
                              
                              # explicit input/output features per architecture
                              model_features={"UED": ModelFeatures(input_cat=["Activity",
                                                                              "Resource"],
                                                                   
                                                                   input_num=["case_elapsed_time",
                                                                              # "event_elapsed_time",
                                                                              "day_in_week",
                                                                              "seconds_in_day"],
                                                                   
                                                                   static_cat=["VariantIndex",
                                                                               "seriousness",
                                                                               "customer",
                                                                               "product",
                                                                               "responsible_section",
                                                                               "seriousness_2",
                                                                               "service_level",
                                                                               "service_type",
                                                                               "support_section",
                                                                               "workgroup"],
                                                                   
                                                                   static_num=[],
                                        
                                                                   # must match with the dynamic attributes used for decision mining (decoding)
                                                                   output_cat=["Activity",
                                                                               "Resource"
                                                                              ],
                                                                    
                                                                   output_num=["case_elapsed_time"
                                                                              ],
                                                                    
                                                                   use_statics=True),
                                            
                                              # has no encoder decoder strcuture but naming is used similar: Only one LSTm with input and output
                                              "FS": ModelFeatures(input_cat=["Activity",
                                                                             "Resource"],
                                                                  
                                                                  input_num=["case_elapsed_time",
                                                                             "day_in_week",
                                                                             "seconds_in_day"
                                                                            ],
                                                              
                                                                  # must match decision mining (instead of activity)
                                                                  output_cat=["Activity",
                                                                              "Resource"
                                                                             ],
                                                                  
                                                                  output_num=["case_elapsed_time"
                                                                             ],
                                                                  
                                                                  use_statics=False),
                                            
                                              "GAN": ModelFeatures(input_cat=["Activity",
                                                                              "Resource"],
                                                                   
                                                                   input_num=["case_elapsed_time",
                                                                              "day_in_week",
                                                                              "seconds_in_day"],
                                                                   
                                                                   # must match decision mining (instead of activity)
                                                                   output_cat=["Activity",
                                                                               "Resource"
                                                                              ],
                                                                   
                                                                   output_num=["case_elapsed_time"
                                                                              ],

                                                                   use_statics=False)}),

    # Sepsis dataset
    "Sepsis": DatasetConfig(key="Sepsis",
                            slug="sepsis",
                            
                            # dynamic attributes (to predict)
                            concept_name="concept:name",
                            
                            dynamic_attributes=["org:group",
                                                # "lifecycle:transition",
                                                "case_elapsed_time",
                                                # "event_elapsed_time",
                                                "Leucocytes",
                                                "CRP",
                                                "LacticAcid"
                                                ],
                            
                            static_attributes=["Age",
                                               "InfectionSuspected",
                                               # "Diagnose",
                                               # "DiagnosticLacticAcid",
                                               # "DiagnosticBlood",
                                               # "DiagnosticArtAstrup",
                                               # "DiagnosticIC",
                                               # "DiagnosticSputum",
                                               # "DiagnosticLiquor",
                                               # "DiagnosticOther",
                                               # "DiagnosticUrinarySediment",
                                               # "DiagnosticECG",
                                               # "DiagnosticUrinaryCulture",
                                               # "DiagnosticXthorax",
                                               # "SIRSCritTachypnea",
                                               # "SIRSCritHeartRate",
                                               "SIRSCriteria2OrMore",  # summary of all?
                                               # "SIRSCritTemperature",
                                               # "SIRSCritLeucos",
                                               # "Hypotensie",
                                               # "Oligurie",
                                               # "Infusion",
                                               # "Hypoxie",
                                               #"DisfuncOrg"
                                               ],
                            # for suffix prediction:
                            event_log=EventLogSpec(event_log_location="data/data/Sepsis.csv",
                                                   # Sepsis has highly variable, concurrent control flow:
                                                   # filter infrequent behaviour so discovery yields a
                                                   # cleaner net. Sweep {0.2, 0.3, 0.4} against the
                                                   # per-place diagnostics (decision_diagnostics).
                                                   net_noise_threshold=0.2,

                                                   case_name="case:concept:name",
                                                   
                                                   timestamp_name="time:timestamp",
                                                   
                                                   cat_dynamic=["concept:name",
                                                                # "lifecycle:transition"
                                                                "org:group",
                                                                ],
                                                   
                                                   num_dynamic=["case_elapsed_time",
                                                                # "event_elapsed_time",
                                                                "day_in_week",
                                                                "seconds_in_day",
                                                                "Leucocytes",
                                                                "CRP",
                                                                "LacticAcid"
                                                                ],
                                                   
                                                   cat_static=["Age",
                                                               "InfectionSuspected",
                                                               # "Diagnose",
                                                               # "DiagnosticLacticAcid",
                                                               # "DiagnosticBlood",
                                                               # "DiagnosticArtAstrup",
                                                               # "DiagnosticIC",
                                                               # "DiagnosticSputum",
                                                               # "DiagnosticLiquor",
                                                               # "DiagnosticOther",
                                                               # "DiagnosticUrinarySediment",
                                                               # "DiagnosticECG",
                                                               # "DiagnosticUrinaryCulture",
                                                               # "DiagnosticXthorax",
                                                               # "SIRSCritTachypnea",          
                                                               # "SIRSCritHeartRate",
                                                               "SIRSCriteria2OrMore",
                                                               # "SIRSCritTemperature",
                                                               # "SIRSCritLeucos",
                                                               # "Hypotensie",
                                                               # "Oligurie",
                                                               # "Infusion",
                                                               # "Hypoxie",
                                                               # "DisfuncOrg"
                                                               ],

                                                    num_static = []),
                            # explicit input/output features per architecture
                            model_features={"UED": ModelFeatures(input_cat=["concept:name",
                                                                            "org:group"],
                                                                 
                                                                 input_num=["case_elapsed_time",
                                                                            "day_in_week",
                                                                            "seconds_in_day",
                                                                            "Leucocytes",
                                                                            "CRP",
                                                                            "LacticAcid"
                                                                            ],
                                                                 
                                                                 static_cat=["Age",
                                                                             "InfectionSuspected",
                                                                             # "Diagnose",
                                                                             # "DiagnosticLacticAcid",
                                                                             # "DiagnosticBlood",
                                                                             # "DiagnosticArtAstrup",
                                                                             # "DiagnosticIC",
                                                                             # "DiagnosticSputum",
                                                                             # "DiagnosticLiquor",
                                                                             # "DiagnosticOther",
                                                                             # "DiagnosticUrinarySediment",
                                                                             # "DiagnosticECG",
                                                                             # "DiagnosticUrinaryCulture",
                                                                             # "DiagnosticXthorax",
                                                                             # "SIRSCritTachypnea",          
                                                                             # "SIRSCritHeartRate",
                                                                             "SIRSCriteria2OrMore",
                                                                             # "SIRSCritTemperature",
                                                                             # "SIRSCritLeucos",
                                                                             # "Hypotensie",
                                                                             # "Oligurie",
                                                                             # "Infusion",
                                                                             # "Hypoxie",
                                                                             # "DisfuncOrg"
                                                                             ],
                                                                 
                                                                 static_num=[],
                                                                 
                                                                 output_cat=["concept:name",
                                                                             "org:group"
                                                                             ],
                                                                 
                                                                 output_num=["case_elapsed_time",
                                                                             "Leucocytes",
                                                                             "CRP",
                                                                             "LacticAcid"
                                                                             ],
                                                                 
                                                                 use_statics=True),
                                
                                             "FS": ModelFeatures(input_cat=["concept:name",
                                                                            "org:group"],
                                                                 
                                                                 input_num=["case_elapsed_time",
                                                                            "day_in_week",
                                                                            "seconds_in_day",
                                                                            "Leucocytes",
                                                                            "CRP",
                                                                            "LacticAcid"
                                                                            ],
                                                                 
                                                                 output_cat=["concept:name",
                                                                             "org:group"], 
                                                                 
                                                                 output_num=["case_elapsed_time",
                                                                             "Leucocytes",
                                                                             "CRP",
                                                                             "LacticAcid"],
                                                                 
                                                                 use_statics=False),
                                
                                             "GAN": ModelFeatures(input_cat=["concept:name",
                                                                             "org:group"],
                                                                  
                                                                  input_num=["case_elapsed_time",
                                                                             "day_in_week",
                                                                             "seconds_in_day",
                                                                             "Leucocytes",
                                                                             "CRP",
                                                                             "LacticAcid"
                                                                             ],
                                                                  
                                                                  output_cat=["concept:name",
                                                                              "org:group"], 
                                                                  
                                                                  output_num=["case_elapsed_time",
                                                                              "Leucocytes",
                                                                              "CRP",
                                                                              "LacticAcid"
                                                                             ],
                                                                  
                                                                  use_statics=False)}),

    # Artificial Prrocurement dataset
    # for dec. mining
    "Procurement": DatasetConfig(key="Procurement", 
                                 slug="procurement",
                                 
                                 concept_name="concept:name",
                                 
                                 dynamic_attributes=["org:resource",
                                                     "case_elapsed_time",
                                                     # "event_elapsed_time",
                                                     "amount",
                                                     "budget_status",
                                                     # "revision_count",
                                                     "supplier_type",
                                                     "goods_match",
                                                     "invoice_deviation_pct",
                                                     ],
                                 
                                 static_attributes=["requester_seniority",
                                                    "department",
                                                    "category", 
                                                    "priority"
                                                    ],
                                 
                                 # for suffix prediction:
                                 event_log=EventLogSpec(event_log_location="data/data/procurement_event_log.csv",
                                                        
                                                        case_name="case:concept:name",
                                                        
                                                        timestamp_name="time:timestamp",
                                                        date_format="%Y-%m-%d %H:%M:%S.%f",
                                                        
                                                        cat_dynamic=["concept:name",
                                                                     "org:resource",
                                                                     "budget_status",
                                                                     "supplier_type",
                                                                     "goods_match"],
                                                        
                                                        num_dynamic=["case_elapsed_time",
                                                                     "event_elapsed_time",
                                                                     "day_in_week",
                                                                     "seconds_in_day",
                                                                     "amount",
                                                                     # "revision_count",
                                                                     "invoice_deviation_pct"],
                                                        
                                                        cat_static=["requester_seniority",
                                                                    "department",
                                                                    "category",
                                                                    "priority"
                                                                    ],

                                                        num_static = []),
                                 # explicit input/output features per architecture
                                 model_features={"UED": ModelFeatures(input_cat=["concept:name",
                                                                                 "org:resource",
                                                                                 "budget_status",
                                                                                 "supplier_type",
                                                                                 "goods_match"
                                                                                 ],
                                                                      
                                                                      input_num=["case_elapsed_time",
                                                                                 "day_in_week",
                                                                                 "seconds_in_day",
                                                                                 "amount",
                                                                                 # "revision_count",
                                                                                 "invoice_deviation_pct"
                                                                                 ],
                                                                      
                                                                      static_cat=["requester_seniority",
                                                                                  "department",
                                                                                  "category",
                                                                                  "priority"],
                                                                      
                                                                      static_num=[],
                                                                      
                                                                      output_cat=["concept:name",
                                                                                  "org:resource",
                                                                                  "budget_status",
                                                                                  "supplier_type",
                                                                                  "goods_match"
                                                                                  ],
                                                                      
                                                                      output_num=["case_elapsed_time",
                                                                                  "amount",
                                                                                  # "revision_count",
                                                                                  "invoice_deviation_pct"
                                                                                  ],
                                                                      
                                                                      use_statics=True),
                                     
                                                 "FS": ModelFeatures(input_cat=["concept:name",
                                                                                "org:resource",
                                                                                "budget_status",
                                                                                "supplier_type",
                                                                                "goods_match"
                                                                                ],
                                                                     
                                                                     input_num=["case_elapsed_time",
                                                                                "day_in_week",
                                                                                "seconds_in_day",
                                                                                "amount",
                                                                                # "revision_count",
                                                                                "invoice_deviation_pct"
                                                                                ],
                                                                     
                                                                     output_cat=["concept:name",
                                                                                 "org:resource",
                                                                                 "budget_status",  
                                                                                 "supplier_type",
                                                                                 "goods_match"
                                                                                 ],
                                                                     output_num=["case_elapsed_time",
                                                                                 "amount",
                                                                                 # "revision_count",
                                                                                 "invoice_deviation_pct"
                                                                                 ],
                                                                     use_statics=False),
                                     
                                                 "GAN": ModelFeatures(input_cat=["concept:name",
                                                                                 "org:resource",
                                                                                 "budget_status",
                                                                                 "supplier_type",
                                                                                 "goods_match"
                                                                                ],
                                                                     
                                                                     input_num=["case_elapsed_time",
                                                                                "day_in_week",
                                                                                "seconds_in_day",
                                                                                "amount",
                                                                                # "revision_count",
                                                                                "invoice_deviation_pct"
                                                                                ],
                                                                     
                                                                     output_cat=["concept:name",
                                                                                 "org:resource",
                                                                                 "budget_status", 
                                                                                 "supplier_type",
                                                                                 "goods_match"
                                                                                 ],
                                                                     output_num=["case_elapsed_time",
                                                                                 "amount",
                                                                                 # "revision_count",
                                                                                 "invoice_deviation_pct"
                                                                                 ],
                                                                     use_statics=False)}),
    
    # BPIC2020 domestic declarations
    "BPIC20_DD": DatasetConfig(key="BPIC20_DD",
                               slug="bpic20_dd",
                               # activity
                               concept_name="concept:name",
                               
                               dynamic_attributes=["org:resource",
                                                   # "org:role",
                                                   "case_elapsed_time",
                                                   # "event_elapsed_time",
                                                   # "day_in_week",
                                                   # "seconds_in_day"
                                                   ],
                               
                               static_attributes=[# "case:BudgetNumber",
                                                  # "case:DeclarationNumber",
                                                  "case:Amount"
                                                 ],
                                
                               event_log=EventLogSpec(event_log_location="data/data/DomesticDeclarations_cleaned.csv",
                                                      raw_source_location="data/data/DomesticDeclarations.csv",
                                                      clean_activity_by_suffix=True,
                                                      # Near-sequential approval flow with infrequent
                                                      # correction/reject variants: a mild filter trims
                                                      # the spurious decision places.
                                                      net_noise_threshold=0.15,
                                                       
                                                      case_name="case:concept:name",
                                                      timestamp_name="time:timestamp",
                                                        
                                                      cat_dynamic=["concept:name",
                                                                   "org:resource",
                                                                   # "org:role"
                                                                  ],
                                                       
                                                      cat_static=[# "case:BudgetNumber",
                                                                  # "case:DeclarationNumber"
                                                                 ],
                                                       
                                                      num_dynamic=["case_elapsed_time",
                                                                   # "event_elapsed_time",
                                                                   "day_in_week",
                                                                   "seconds_in_day"
                                                                  ],
                                                       
                                                      num_static=["case:Amount"]),
                               
                               # explicit input/output features per architecture
                               model_features={"UED": ModelFeatures(input_cat=["concept:name",
                                                                               "org:resource",
                                                                               ],
                                                                    
                                                                    input_num=["case_elapsed_time",
                                                                               "day_in_week",
                                                                               "seconds_in_day"
                                                                               ],
                                                                    
                                                                    static_cat=[#"case:BudgetNumber"
                                                                               ],
                                                                    
                                                                    static_num=["case:Amount"],
                                                                    
                                                                    output_cat=["concept:name",
                                                                                "org:resource"
                                                                                ],
                                                                    
                                                                    output_num=["case_elapsed_time",
                                                                               ],
                                                                    
                                                                    use_statics=True),
                                   
                                               "FS": ModelFeatures(input_cat=["concept:name",
                                                                              "org:resource"
                                                                              ],
                                                                    
                                                                   input_num=["case_elapsed_time",
                                                                              "day_in_week",
                                                                              "seconds_in_day"
                                                                              ],
                                                                    
                                                                   output_cat=["concept:name",
                                                                               "org:resource"],
                                                                    
                                                                   output_num=["case_elapsed_time"
                                                                              ],
                                                                    
                                                                   use_statics=False),
                                                
                                                "GAN": ModelFeatures(input_cat=["concept:name",
                                                                                "org:resource"],
                                                                    
                                                                     input_num=["case_elapsed_time",
                                                                                "day_in_week",
                                                                                "seconds_in_day"
                                                                                ],
                                                                    
                                                                     output_cat=["concept:name",
                                                                                "org:resource"],  
                                                                     
                                                                     output_num=["case_elapsed_time"
                                                                                ],
                                                                     use_statics=False)})}


# Initialization of model, hyperparams and decoding modes
MODELS: Dict[str, ModelConfig] = {
    # Uncertainty-aware encoder decoder LSTM
    "UED": ModelConfig(key="UED",
                       model_file="UED_LSTM",
                       model_slug="ued_lstm",
                       
                       # model hypers train
                       hidden_size=128,
                       num_layers=4,
                       dropout=0.1,
                       learning_rate=1e-5,

                       # decision hypers train
                       lambda_sem=0.3,
                       tau=0.2,

                       extra={"optimizer": "adam",
                              # for standard losses:
                              "weight_decay": 0.0,
                              # for loss attenuation:
                              "regularization_term": 1e-4,
                              # teacher forcing
                              "teacher_forcing_mode": "scheduled", "min_teacher_forcing_value": 0.0, "max_teacher_forcing_value": 1.0,
                              # also model static attributes in a seperated layes                              
                              "use_statics": True,
                              # Evaluation: probabilistic MC sampling (plain) / mcsa (guided).
                              "decode_mode": "probabilistic",
                              "guided_kind": "mcsa",
                              "is_probabilistic": True,
                              
                              # dropout durng evaluation
                              "eval_dropout": 0.1}),
    
    # FS-LSTM (FullShared_Join_LSTM): next-event prediction, single step, no TF.
    "FS": ModelConfig(key="FS",
                      model_file="FS_LSTM",
                      model_slug="fs_lstm",

                      hidden_size=50,
                      num_layers=1,
                      learning_rate=1e-3,
                      
                      # semantic loss decision train hypers
                      lambda_sem=0.5,
                      tau=0.2,

                      extra={"optimizer": "adam",
                             "weight_decay": 0.0,
                             "input_size": 1,
                             
                             # Evaluation: arg-max (plain) / mode (guided), deterministic.
                             "decode_mode": "mode",
                             "guided_kind": "mode",
                             "is_probabilistic": False}),
    
    # GAN-LSTM (TaymouriAdversarialLSTM): GAN (MLMME) + Gumbel-softmax.
    "GAN": ModelConfig(key="GAN",
                       model_file="GAN_LSTM",
                       model_slug="gan_lstm",

                     hidden_size=200,
                     num_layers=5,
                     dropout=0.2,

                     # decision aware train values
                     lambda_sem=0.3,
                     tau=0.2,

                     learning_rate=5e-5,

                     extra={"optimizer": "rmsprop",
                            "input_size": 1,
                            "teacher_forcing_mode": "scheduled",
                            "min_teacher_forcing_value": 0.0,
                            "max_teacher_forcing_value": 1.0,
                            "tau_start": 0.9,
                            "tau_min": 0.01,
                            "use_gan": True,
                            "beam_width": 3,
                            
                            # Evaluation: beam search (plain) / beam (guided), deterministic.
                            # can this be implemented clearer and easier??
                            "decode_mode": "beam",
                            "guided_kind": "beam",
                            "is_probabilistic": False})}

# Decision-guided decoding defaults (Helpdesk guided eval used these; confirm per dataset before relying on them elsewhere).
@dataclass(frozen=True)
class GuidanceConfig:
    epsilon: float = 1e-3
    beta_max: float = 2.0
    alpha: float = 0.10
    support_threshold: float = 0.05
    samples_per_case: int = 100

    # Confidence / observability gating of the guided reweighting 
    # (all no-ops by default; see decision_rule_guided_reasoning_inference.DecisionGuidanceConfig).
    #   min_base_entropy        (a) skip guidance when the base next-event dist is
    #                               already peaked (normalised entropy < this).
    #   min_decision_confidence (b) skip guidance when c_i = max_a z_i(a) < this.
    #   max_guided_steps        (c) only guide while decode step index <= this
    #                               (context still from the observed prefix).
    min_base_entropy: float = 0.0
    min_decision_confidence: float = 0.0
    max_guided_steps: Optional[int] = None


# Experiment + path derivation
@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig
    model: ModelConfig
    variant: Variant
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    probabilistic_samples: int = 100
    num_processes: int = 32

@dataclass(frozen=True)
class ExperimentPaths:
    project_root: Path
    test_dataset: Path
    train_dataset: Path
    val_dataset: Path
    model_checkpoint: Path
    eval_outputs: Path
    eval_reasoning: Path          # only meaningful for guided decode
    petri_net: Path
    decision_bundle: Path
    decision_model_dir: Path
    numeric_scalers: Path

@dataclass(frozen=True)
class DatasetPaths:
    """
    Paths for building a dataset's artifacts (model/variant-independent).
    """
    project_root: Path
    raw_event_log: Path
    raw_source_log: Optional[Path]      
    raw_prefix_dir: Path
    petri_net_pkl: Path
    petri_net_png: Path
    normal_dir: Path                    
    decision_labeled_dir: Path          
    decision_bundle: Path
    decision_model_dir: Path
    numeric_scalers: Path

    def _stem(self, ds: "DatasetConfig", split: str) -> str:
        return f"{ds.result_name}_{ds.event_log.min_suffix_size}_{split}"

    def raw_prefix_csv(self, ds: "DatasetConfig", split: str) -> Path:
        return self.raw_prefix_dir / f"{self._stem(ds, split)}.csv"

    def normal_tensor(self, ds: "DatasetConfig", split: str) -> Path:
        return self.normal_dir / f"{self._stem(ds, split)}.pkl"

    def decision_tensor(self, ds: "DatasetConfig", split: str) -> Path:
        return self.decision_labeled_dir / f"{self._stem(ds, split)}.pkl"


def resolve_dataset_paths(ds: DatasetConfig, root: Optional[Path] = None) -> DatasetPaths:
    """
    Derive the data-pipeline paths for a dataset (used by data_loading).
    """
    # The raw event log lives in the shared data store, reached via the manual
    # relative root (set intentionally; resolves to /home/PSPLab/data from
    # src/notebooks). Everything *generated* lives in the repo, reached via
    # PROJECT_ROOT (the same absolute root resolve_paths uses for train/eval).
    raw_root = Path("../../../../")
    root = (root or PROJECT_ROOT).resolve()

    data = root / "data" / ds.key
    
    petri_dir = data / "Petri_net"
    
    daw = petri_dir / "data_aware_Petri_net"
    
    src_log = ds.event_log.raw_source_location
    
    return DatasetPaths(project_root=root,
                        #
                        raw_event_log=raw_root / ds.event_log.event_log_location,
                        #
                        raw_source_log=(raw_root / src_log) if src_log else None,
                        #
                        raw_prefix_dir=data / "raw_data",
                        #
                        petri_net_pkl=petri_dir / f"{ds.slug}.pkl",
                        #
                        petri_net_png=petri_dir / f"{ds.slug}.png",
                        #
                        normal_dir=data / "tensor_data" / "normal",
                        #
                        decision_labeled_dir=data / "tensor_data" / "decision_labeled",
                        #
                        decision_bundle=daw / "decision_places_bundle.json",
                        #
                        decision_model_dir=daw / "models",
                        #
                        numeric_scalers=daw / "numeric_scalers.pkl")


def check_model_features(dataset: str | DatasetConfig,
                         strict: bool = False) -> Dict[str, Dict[str, object]]:
    """
    Sanity-check each architecture's input/output feature lists against the
    decision-mining attributes, and report the decoding "match": which
    decision-mining DYNAMIC attributes the model PREDICTS (present in output_* ->
    fed to the decision model as predicted values during guided decoding) versus
    which are CARRIED FORWARD from the prefix's last event (not predicted by the
    model). Static decision attributes always come from the constant case-level
    prefix, so they are always available.

    All three architectures predict the non-activity dynamic attributes that are
    in their feature set (UED via its decoder, FS / GAN via auxiliary heads), so a
    decision attribute is predicted exactly when it appears in output_*.

    Always raises ValueError if the activity (concept_name) is missing from a
    model's input_cat or output_cat. When ``strict=True`` also raises if any model
    fails to predict a decision-mining dynamic attribute (a genuine mismatch:
    the decision model would need a value the suffix model never produces).
    Returns ``{model_key: {...}}`` for inspection.
    """
    ds = DATASETS[dataset] if isinstance(dataset, str) else dataset
    concept = ds.concept_name
    dyn = [a for a in ds.dynamic_attributes if a != concept]
    report: Dict[str, Dict[str, object]] = {}
    mismatches: Dict[str, List[str]] = {}
    for key, fs in ds.model_features.items():
        if concept not in fs.input_cat:
            raise ValueError(f"{ds.key}/{key}: activity '{concept}' missing from input_cat")
        if concept not in fs.output_cat:
            raise ValueError(f"{ds.key}/{key}: activity '{concept}' missing from output_cat")
        predicted = set(fs.output_cat) | set(fs.output_num)
        carried = [a for a in dyn if a not in predicted]
        report[key] = {"predicted_decision_dyn": [a for a in dyn if a in predicted],
                       "carried_forward_decision_dyn": carried,
                       "static_decision_attrs": list(ds.static_attributes)}
        if carried:
            mismatches[key] = carried
    if strict and mismatches:
        detail = "; ".join(f"{ds.key}/{k} does not predict {v}" for k, v in mismatches.items())
        raise ValueError(
            "Decision-attribute mismatch: the decision model requires dynamic event "
            f"attributes that the suffix-prediction model does not predict ({detail}). "
            "Add them to that model's ModelFeatures.output_* (and input_*) so the "
            "predicted values are available to the decision model during decoding.")
    return report


def require_predicted_decision_attrs(dataset: str | DatasetConfig, model: str) -> None:
    """
    Raise ValueError if ``model`` does not predict every decision-mining dynamic
    attribute for ``dataset`` (i.e. some are carried-forward rather than
    predicted). Used to gate the decision-model paths (guided decode, conformance,
    decision-aware training), so a mismatch fails fast with a clear message.
    """
    ds = DATASETS[dataset] if isinstance(dataset, str) else dataset
    report = check_model_features(ds)  # validates activity presence
    info = report.get(model)
    if info is None:
        raise KeyError(f"{ds.key}: no ModelFeatures for model '{model}'")
    missing = info["carried_forward_decision_dyn"]
    if missing:
        raise ValueError(
            f"Decision-attribute mismatch for {ds.key}/{model}: the decision model "
            f"requires dynamic event attributes {missing} that this suffix-prediction "
            "model does not predict. Add them to its ModelFeatures.output_* (and "
            "input_*) so the predicted values reach the decision model during decoding.")


def make_experiment(dataset: str, model: str, variant: str | Variant,
                    **overrides) -> ExperimentConfig:
    """
    Look up registry entries by key and build an ExperimentConfig.
    """
    ds = DATASETS[dataset]
    md = MODELS[model]
    var = Variant(variant)
    return ExperimentConfig(dataset=ds, model=md, variant=var, **overrides)

def resolve_paths(cfg: ExperimentConfig, root: Optional[Path] = None) -> ExperimentPaths:
    """
    Derive every concrete file path from the registry + naming conventions.
    """
    root = (root or PROJECT_ROOT).resolve()
    ds, md, var = cfg.dataset, cfg.model, cfg.variant

    data = root / "data" / ds.key
    tensor = data / "tensor_data"
    train_subdir = "decision_labeled" if var.uses_decision_loss else "normal"

    model_subdir = var.model_source                       
    model_tag = "DA" if var.model_source == "decision" else "clean"

    cache_tag = "decision_guided" if var.decode == "guided" else var.model_source
    eval_dir = root / "eval_results" / ds.key / var.value
    cache_stem = f"{ds.slug}_{md.model_slug}_{cache_tag}"

    petri_dir = data / "Petri_net"
    daw = petri_dir / "data_aware_Petri_net"

    sfx = ds.event_log.min_suffix_size

    return ExperimentPaths(project_root=root,
                           #
                           test_dataset=tensor / "normal" / f"{ds.result_name}_{sfx}_test.pkl",
                           #
                           train_dataset=tensor / train_subdir / f"{ds.result_name}_{sfx}_train.pkl",
                           # 
                           val_dataset=tensor / train_subdir / f"{ds.result_name}_{sfx}_val.pkl",
                           #
                           model_checkpoint=root / "models" / ds.key / model_subdir/ f"{ds.key}_{md.model_file}_v1_{model_tag}.pkl",
                           #
                           eval_outputs=eval_dir / f"{cache_stem}_outputs.pkl",
                           #
                           eval_reasoning=eval_dir / f"{cache_stem}_reasoning.pkl",
                           #
                           petri_net=petri_dir / f"{ds.slug}.pkl",
                           #
                           decision_bundle=daw / "decision_places_bundle.json",
                           #
                           decision_model_dir=daw / "models",
                           #
                           numeric_scalers=daw / "numeric_scalers.pkl")
