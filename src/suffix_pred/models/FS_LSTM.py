import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F

class FullShared_Join_LSTM(nn.Module):
    def __init__(self,
                 data_set_categories : list[tuple[str, dict[str, int]]],
                 hidden_size: int,
                 num_layers: int,
                 model_feat: list,
                 input_size:int,
                 output_size_act: int):

        super().__init__()

        # Feature sizes
        self.data_set_categories = data_set_categories
        print("Data set categories: ", data_set_categories)

        self.model_feat = model_feat
        print("Model input features: ", model_feat)

        print("\n")

        # List containing two dicts: One for categorical, one for numerical
        cat_categories, num_categories = data_set_categories

        cat_input_feat_model, num_input_feat_model = model_feat

        # Use the first value in the tuple as the key and the second value as the value
        cat_dict = {cat[0]: cat[1] for cat in cat_categories}

        # Get input feature sizes to determine the model input size
        list_of_classes_per_cat = [cat_dict[cat_feat] for cat_feat in cat_input_feat_model if cat_feat in cat_dict]

        # Track the activity feature so we can label the activity-head output specially.
        # Activity is the first categorical feature that has 'concept' and 'name' or 'activity' in it,
        # otherwise the first categorical feature.
        self._activity_feature_name = self._select_activity_feature(cat_input_feat_model)

        # Non-activity dynamic features that we will also predict.
        self._cat_feature_names = list(cat_input_feat_model)
        self._num_feature_names = list(num_input_feat_model)
        self._other_cat_feature_names = [name for name in self._cat_feature_names
                                         if name != self._activity_feature_name and name in cat_dict]
        self._other_cat_class_counts = {name: cat_dict[name] for name in self._other_cat_feature_names}

        # Create embeddings for categorical features
        self.embeddings = nn.ModuleList([nn.Embedding(n_cat, min(600, round(1.6 * n_cat**0.56))) for n_cat in list_of_classes_per_cat])
        print("Embeddings: ", self.embeddings)

        # Compute total input size encoder
        embedding_size = sum([min(600, round(1.6 * n_cat**0.56)) for n_cat in list_of_classes_per_cat])
        print("Total embedding feature size: ", embedding_size)

        # Only add embedding to input size in case of training. When model is saved,
        # the input size expected is already correct.
        if input_size == 1:
            self.input_size = len(num_input_feat_model) + embedding_size
        else:
            self.input_size = input_size
        print("Input feature size: ", self.input_size)

        self.hidden_size = hidden_size
        print("Cells hidden size: ", hidden_size)

        self.num_layers = num_layers
        print("Number of LSTM layer: ", num_layers)

        self.output_size_act = output_size_act

        # Shared LSTM layer
        self.shared_lstm = nn.LSTM(input_size=self.input_size,
                                   hidden_size=self.hidden_size,
                                   batch_first=True,
                                   dropout=0.2,
                                   num_layers=self.num_layers)

        # batch‐norm across features, per time‐step (permute (B, T, hidden_size) -> (B, H, T) for BatchNorm1d, then back)
        self.bn1 = nn.BatchNorm1d(self.hidden_size)

        # Separate prediction head per output variable. Every output (activity,
        # each non-activity categorical attribute and each numerical attribute)
        # gets its OWN LSTM head operating on the shared representation, followed
        # by its own linear layer. This way each output predicts and continues
        # through a fully separated prediction head instead of sharing the
        # activity head's hidden state.
        self.lstm_act = self._make_head_lstm()

        # Linear output heads:
        self.act_head  = nn.Linear(self.hidden_size, self.output_size_act)

        # Separate LSTM + linear head per non-activity categorical feature, so we
        # can roll the prefix forward at inference using predicted values instead
        # of ground truth.
        self.other_cat_head_lstms = nn.ModuleDict()
        self.other_cat_heads = nn.ModuleDict()
        for feat_name, n_classes in self._other_cat_class_counts.items():
            self.other_cat_head_lstms[feat_name] = self._make_head_lstm()
            self.other_cat_heads[feat_name] = nn.Linear(self.hidden_size, n_classes)

        # Separate LSTM + linear head per numerical feature.
        self.num_head_lstms = nn.ModuleDict()
        self.num_heads = nn.ModuleDict()
        for feat_name in self._num_feature_names:
            self.num_head_lstms[feat_name] = self._make_head_lstm()
            self.num_heads[feat_name] = nn.Linear(self.hidden_size, 1)

    def _make_head_lstm(self):
        """Build a dedicated LSTM prediction head over the shared representation."""
        return nn.LSTM(input_size=self.hidden_size,
                       hidden_size=self.hidden_size,
                       batch_first=True,
                       dropout=0.2,
                       num_layers=self.num_layers)

    @staticmethod
    def _select_activity_feature(cat_input_feat_model):
        for name in cat_input_feat_model:
            lowered = str(name).lower()
            if ("concept" in lowered and "name" in lowered) or "activity" in lowered:
                return name
        return cat_input_feat_model[0] if cat_input_feat_model else None
    
    def __input_construction(self, data):
        cats, nums = data
                
        # Embedd categorical tensors
        embedded_cats = []
        for i, embedd in enumerate(self.embeddings):        
            embedded_cats.append(embedd(cats[i]))

        # Merged categroical data
        merged_cats = torch.cat([cat for cat in embedded_cats], dim=-1)
        
        if len(nums):
            # Merged numerical inputs
            merged_nums = torch.cat([num.unsqueeze(2) for num in nums], dim=-1)
        else:
            merged_nums = torch.tensor([], device=merged_cats.device)

        # Merged input
        x = torch.cat((merged_cats, merged_nums), dim=-1).permute(1,0,2) # dim: seq_len x batch_size x input_features
        
        return x

    def forward(self, input, return_dict: bool = False):
        # Build your input: x of shape (T = seq. len., B = batch size, input_size)
        x = self.__input_construction(data=input)      # (T, B, input_size)

        x = x.permute(1, 0, 2) # (B, T, input_size)

        # Shared LSTM (batch_first=True)
        out_seq, _ = self.shared_lstm(x) # (B, T, hidden_size)

        # Batch‑norm over features at each time step
        y = out_seq.transpose(1, 2) # (B, hidden_size, T)
        y = self.bn1(y) # (B, hidden_size, T)
        y = y.transpose(1, 2) # (B, T, hidden_size)

        # Activity head: own LSTM (batch_first=True); grab only last hidden state.
        self.lstm_act.flatten_parameters()
        _, (h_act,  _) = self.lstm_act(y) # h_act: (num_layers, B, hidden_size)
        # Use the last layer's hidden state if multi-layer.
        h_act_last = h_act[-1]  # (B, hidden_size)

        # Final heads & activations
        a_logits = self.act_head(h_act_last)  # (B, output_size_act)

        if not return_dict:
            return a_logits

        # Auxiliary predictions for non-activity dynamic attributes so the caller
        # can roll the prefix forward at inference without GT leakage. Each output
        # runs through its own separated LSTM head over the shared representation.
        other_cat_logits = {}
        for feat, head_lstm in self.other_cat_head_lstms.items():
            head_lstm.flatten_parameters()
            _, (h_feat, _) = head_lstm(y)
            other_cat_logits[feat] = self.other_cat_heads[feat](h_feat[-1])

        num_means = {}
        for feat, head_lstm in self.num_head_lstms.items():
            head_lstm.flatten_parameters()
            _, (h_feat, _) = head_lstm(y)
            num_means[feat] = self.num_heads[feat](h_feat[-1]).squeeze(-1)

        return {"activity_logits": a_logits,
                "other_cat_logits": other_cat_logits,
                "num_means": num_means}

    def predict_next_event(self, input):
        """
        Convenience inference helper returning a dict of predicted next-event values.

        Output keys:
        - "activity_id": Long tensor [B] of the argmax activity id
        - "cat_ids": dict {feature_name: Long tensor [B]} for non-activity categorical features
        - "num_values": dict {feature_name: Float tensor [B]} for numerical features
        """
        was_training = self.training
        if was_training:
            self.eval()
        try:
            with torch.no_grad():
                out = self.forward(input, return_dict=True)
        finally:
            if was_training:
                self.train()

        activity_ids = torch.argmax(out["activity_logits"], dim=-1)
        cat_ids = {feat: torch.argmax(logits, dim=-1) for feat, logits in out["other_cat_logits"].items()}
        num_values = {feat: mean.detach() for feat, mean in out["num_means"].items()}
        return {"activity_id": activity_ids,
                "cat_ids": cat_ids,
                "num_values": num_values}
    
    def save(self, path : str):
        """
        Store the trained model at path.
        """
        checkpoint = {'model_state_dict' : self.state_dict(),
                      'kwargs' : 
                          {'data_set_categories' : self.data_set_categories,
                           'hidden_size': self.hidden_size,
                           'num_layers': self.num_layers,
                           'model_feat': self.model_feat,
                           'input_size': self.input_size,
                           'output_size_act': self.output_size_act}}
        return torch.save(checkpoint, path)

    @staticmethod
    def load(path : str):
        """
        Load the stored model at path
        """
        checkpoint = torch.load(path, weights_only=False, map_location=torch.device("cpu"))
        model = FullShared_Join_LSTM(**checkpoint['kwargs'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
