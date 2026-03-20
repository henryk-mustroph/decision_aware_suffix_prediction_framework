"""
According to:
M. Kunkler, H. Mustroph and S. Rinderle-Ma, "Probabilistic Suffix Prediction of Business Processes, International Conference on Process Mining (ICPM) 2025, doi: 10.1109/ICPM66919.2025.11220650.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"
import torch
from torch import Tensor, nn
from typing import List, Optional, Tuple, Union

class DropoutUncertaintyEncoderDecoderLSTM(nn.Module):
    """
    Full Encoder-Decoder architecture with droput uncertainty LSTM.
    """

    def __init__(self,
                 data_set_categories: list[tuple[str, dict[str, int]]],
                 enc_feat: list,
                 dec_feat: list,
                 seq_len_pred: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float,
                 # optional static attributes
                 static_data_set_categories: Optional[list[tuple[str, dict[str, int]]]] = None,
                 static_enc_feat: Optional[list] = None):
        """
        Full Encoder-Decoder architecture with droput uncertainty LSTM.

        Args:
        - data_set_categories: Event attributes, name and size
        - enc_feat: Event attributes used by encoder as input
        - dec_feat: Event attributes used by decoder as input and output
        - seq_len_pred: Length of the predicted suffix sequence
        - hidden_size: Hidden size for LSTM cells and fully connected layers
        - num_layers: Number of hidden layers in both Encoder and Decoder
        - dropout (float): Dropout probability, must be in [0, 1). Required.
        - static_data_set_categories: Event attribute categories for static encoder input
        - static_enc_feat: Static event attributes used by encoder as input
        """
        super(DropoutUncertaintyEncoderDecoderLSTM, self).__init__()

        # Feature sizes encoder
        self.data_set_categories = data_set_categories
        print("Dynamic data set categories: ", data_set_categories)
        self.static_data_set_categories = static_data_set_categories
        print("Data set static categories: ", static_data_set_categories)

        self.enc_feat = enc_feat
        print("Encoder dynamic input features: ", enc_feat)

        self.static_enc_feat = static_enc_feat
        if self.static_enc_feat:
            print("Encoder static input features: ", self.static_enc_feat)

        self.dec_feat = dec_feat
        print("Decoder input and output features: ", dec_feat)
        # Sequence lenght prediciton
        self.seq_len_pred = seq_len_pred
        print("Sequence length of decoder output: ", seq_len_pred)

        print("\n")

        # Parameters for encoder and decoder
        self.hidden_size = hidden_size
        print("LSTM cells and FC hidden size: ", hidden_size)
        self.num_layers = num_layers
        print("Number of LSTM layer: ", num_layers)
        self.dropout = dropout
        print("Dropout rate: ", dropout)

        print("\n")

        # Encoder (dynamic)
        print("Encoder dynamic:")
        # Get list of category label values for cat and num
        enc_label_cats, enc_label_nums = self.__get_list_labels_input(data_set_categories=data_set_categories, model_type_feats=enc_feat)
        self.data_labels_features_enc = [enc_label_cats, enc_label_nums]
        print("Encoder number of labels for each input feature (categorical, numerical):", self.data_labels_features_enc)
        
        data_cat_indices_enc, data_num_indices_enc = self.__get_list_tensor_indeces(data_set_categories=data_set_categories, model_type_feats=enc_feat)
        self.data_indices_enc = [data_cat_indices_enc, data_num_indices_enc]
        print("Encoder indices of tensors in dataset used as input:", self.data_indices_enc)

        # Create embeddings for categorical features
        self.embeddings_enc = nn.ModuleList([nn.Embedding(n_cat, min(600, round(1.6 * n_cat**0.56))) for n_cat in enc_label_cats])
        print("Embeddings encoder: ", self.embeddings_enc)

        # Compute total input size encoder
        embedding_size_enc = sum([min(600, round(1.6 * n_cat**0.56)) for n_cat in enc_label_cats])
        print("Total embedding feature size encoder: ", embedding_size_enc)
        num_size_enc = sum(enc_label_nums)
        print("Total numerical feature size encoder: ", num_size_enc)
        self.input_size_enc = embedding_size_enc + num_size_enc
        print("Input feature size encoder: ", self.input_size_enc)

        print("\n")

        # Encoder (static)
        print("Encoder static:")
        if self.static_enc_feat:
            if self.static_data_set_categories is None:
                raise ValueError("Static encoder features provided but static categories aremissing.")
            
            static_enc_label_cats, static_enc_label_nums = self.__get_list_labels_input(data_set_categories=self.static_data_set_categories,
                                                                                        model_type_feats=self.static_enc_feat)
            self.data_labels_static_features_enc = [static_enc_label_cats,
                                                    static_enc_label_nums]
            
            print("Encoder number of labels for each input feature"
                  " (categorical, numerical):",self.data_labels_static_features_enc)
            
            static_data_cat_indices_enc, static_data_num_indices_enc = (self.__get_list_tensor_indeces(data_set_categories=self.static_data_set_categories,
                                                                                                       model_type_feats=self.static_enc_feat)
                                                                        )
            self.static_data_indices_enc = [static_data_cat_indices_enc,
                                            static_data_num_indices_enc]
            print("Encoder indices of tensors in dataset used as input (static):",
                  self.static_data_indices_enc
                  )

            # Embedding
            if static_enc_label_cats:
                self.embeddings_static_enc = nn.ModuleList([nn.Embedding(n_cat, min(600, round(1.6 * n_cat**0.56)))
                                                            for n_cat in static_enc_label_cats])
                static_embedding_size = sum([embedding.embedding_dim
                                             for embedding in self.embeddings_static_enc])
                
                print("Static encoder categorical embeddings:", self.embeddings_static_enc)
            
            else:
                self.embeddings_static_enc = None
                static_embedding_size = 0
            print("Total embedding feature size encoder (static): ", static_embedding_size)

            if static_enc_label_nums:
                static_num_size = sum(static_enc_label_nums)
            else:
                static_num_size = 0
            print("Total numerical feature size encoder (static): ", static_num_size)

            # Total sizes
            self.static_input_size_enc = static_embedding_size + static_num_size
            print("Static encoder feature size: ", self.static_input_size_enc)

            # Define Encoder
            self.encoder = DropoutUncertaintyLSTMEncoder(hidden_size=hidden_size,
                                                         # dynamics
                                                         embeddings=self.embeddings_enc,
                                                         data_indices_enc=self.data_indices_enc,
                                                         input_size=self.input_size_enc,
                                                         # layers
                                                         num_layers=num_layers,
                                                         # static feat
                                                         static_embeddings=self.embeddings_static_enc,
                                                         static_data_indices=self.static_data_indices_enc,
                                                         static_input_size=self.static_input_size_enc,
                                                         # dropout
                                                         dropout=dropout)

        else:
            print("No static encoder features configured.")

            # Define Encoder
            self.encoder = DropoutUncertaintyLSTMEncoder(hidden_size=hidden_size,
                                                         # dynamics
                                                         embeddings=self.embeddings_enc,
                                                         data_indices_enc=self.data_indices_enc,
                                                         input_size=self.input_size_enc,
                                                         # layers
                                                         num_layers=num_layers,
                                                         # dropout
                                                         dropout=dropout,
                                                        )
        print("Encoder initialized! \n")

        # Decoder
        # Get list of category label values for cat and num
        dec_label_cats, dec_label_nums = self.__get_list_labels_input(data_set_categories=data_set_categories, model_type_feats=dec_feat)
        
        print("Decoder label values size for each categorical input feature: ",
              dec_label_cats,)
        
        print("Decoder label values size for each numerical input feature: ",
              dec_label_nums,)

        data_cat_indices_dec, data_num_indices_dec = self.__get_list_tensor_indeces(data_set_categories=data_set_categories, model_type_feats=dec_feat)
        self.data_indices_dec = [data_cat_indices_dec, data_num_indices_dec]
        
        print("Decoder indices of tensors in dataset used as input: ",
              self.data_indices_dec,)

        # Create embeddings for categorical features
        self.embeddings_dec = nn.ModuleList([nn.Embedding(n_cat, min(600, round(1.6 * n_cat**0.56))) for n_cat in dec_label_cats])
        
        print("Embeddings decoder: ", self.embeddings_dec)

        # Compute total input size decoder
        embedding_size_dec = sum([min(600, round(1.6 * n_cat**0.56)) for n_cat in dec_label_cats])
        
        print("Total embedding feature size decoder: ", embedding_size_dec)

        num_size_dec = sum(dec_label_nums)
        print("Total numerical feature size decoder: ", num_size_dec)

        self.input_size_dec = embedding_size_dec + num_size_dec
        print("Input feature size decoder: ", self.input_size_dec)

        # Dictionary of output features and output_sizes
        self.output_sizes = self.__get_list_dict_labels_output(data_set_categories=data_set_categories, model_type_feats=dec_feat)
        
        # Categorical-only setup: keep only categorical output heads
        self.output_sizes[1] = {}
        print("Output feature list of dicts (featue name, feature output size)"
              " of decoder:",
             self.output_sizes,)

        # Define Decoder
        self.decoder = DropoutUncertaintyLSTMDecoder(input_size=self.input_size_dec,
                                                     hidden_size=hidden_size,
                                                     output_sizes=self.output_sizes,
                                                     embeddings=self.embeddings_dec,
                                                     data_indices_dec=self.data_indices_dec,
                                                     num_layers=num_layers,
                                                     dropout=dropout)
        print("Decoder initialized! \n")

        # List containing two dicts: One for categorical, one for numerical
        self.output_feature_indeces = self.__get_list_dict_feature_index(data_set_categories=data_set_categories, model_type_feats=dec_feat)
        
        # Categorical-only setup: keep only categorical output indices
        self.output_feature_indeces[1] = {}

    def __get_list_labels_input(self, data_set_categories, model_type_feats):
        """
        Gets number of feature attributes used as input of model.

        Returns two lists (categorical, numerical) containing the number of feature attributes.
        """
        # Unpack categories
        cat_categories, num_categories = data_set_categories
        cat_feat_model, num_feat_model = model_type_feats

        # Use the first value in the tuple as the key and the second value as the value
        cat_dict = {cat[0]: cat[1] for cat in cat_categories}
        num_dict = {num[0]: num[1] for num in num_categories}

        # Get input feature sizes to determine the model input size
        label_cats = [cat_dict[cat_feat] for cat_feat in cat_feat_model if cat_feat in cat_dict]
        
        label_nums = [num_dict[num_feat] for num_feat in num_feat_model if num_feat in num_dict]

        return label_cats, label_nums

    def __get_list_tensor_indeces(self, data_set_categories, model_type_feats):
        """
        Gets indices of tensors in dataset used as input of model.

        Returns two lists (categorical, numerical) containing the indices of the tensors
        in the datset used as input for the encoder.
        """
        # Unpack categories
        cat_categories, num_categories = data_set_categories
        cat_feat_model, num_feat_model = model_type_feats

        # Convert cat_feat_model and num_feat_model to sets for O(1) membership checks
        cat_feat_set = set(cat_feat_model)
        num_feat_set = set(num_feat_model)

        # Get indices of tensors used as input of model
        cat_indices = [i for i, cat in enumerate(cat_categories) if cat[0] in cat_feat_set]
        
        num_indices = [i for i, num in enumerate(num_categories) if num[0] in num_feat_set]

        return cat_indices, num_indices

    def __get_list_dict_labels_output(self, data_set_categories, model_type_feats):
        """
        Return list of dictionary labels.
        """
        # Unpack categories
        cat_categories, num_categories = data_set_categories
        cat_feat_model, num_feat_model = model_type_feats

        # Use the first value in the tuple as the key and the second value as the value
        cat_dict = {cat[0]: cat[1] for cat in cat_categories}
        num_dict = {num[0]: num[1] for num in num_categories}

        # Create separate dictionaries for categorical and numerical features
        cat_labels_dict = {cat_feat: cat_dict[cat_feat]
                           for cat_feat in cat_feat_model
                           if cat_feat in cat_dict}
        num_labels_dict = {num_feat: num_dict[num_feat]
                           for num_feat in num_feat_model
                           if num_feat in num_dict}

        # Return a list containing two dicts: one for categorical and one for numerical
        return [cat_labels_dict, num_labels_dict]

    def __get_list_dict_feature_index(self, data_set_categories, model_type_feats):
        """
        Gets lisft of dicts of feature names and their tensor indices in dataset.
        """
        # Unpack categories
        cat_categories, num_categories = data_set_categories
        cat_feat_model, num_feat_model = model_type_feats

        # Convert cat_feat_model and num_feat_model to sets for O(1) membership checks
        cat_feat_set = set(cat_feat_model)
        num_feat_set = set(num_feat_model)

        # Create dictionaries to store feature names and their index positions
        cat_index_dict = {cat[0]: i for i, cat in enumerate(cat_categories) if cat[0] in cat_feat_set}
        num_index_dict = {num[0]: i for i, num in enumerate(num_categories) if num[0] in num_feat_set}

        # Return a list of two dicts: one for categorical and one for numerical features
        return [cat_index_dict, num_index_dict]

    def forward(self,
                prefixes: List,
                static_inputs: Optional[Union[Tensor, List, Tuple, dict]] = None,
                suffixes: Optional[List] = None,
                teacher_forcing_ratio: Optional[float] = 0.0,
                prefix_mask: Optional[Tensor] = None,
                return_teacher_forcing_mask: Optional[bool] = False):
        """
        Full forward pass through the Encoder-Decoder architecture.

        Inputs:
        - prefixes: Input prefix sequence:
        - static_inputs: Optional static attribute tensor(s) aligned with
        - suffixes: Suffix to predict:
        - teacher_forcing_ratio: Value between 0 and 1 to select pred or target as last event.
        - prefix_mask: masking of zero padding for prefix for encoder.

        Outputs:
        - predictions: Predicted outcome.
        - (h,c): Predicted last hidden and cell state.
        - self.seq_len_pred: Sequence length.
        - self.output_feature_indeces: Target data indices:
        """
        # Model is in training mode and suffixes are provided
        training = self.training and suffixes is not None
        # Model is in evaluation (validation) mode and suffixes are provided
        validation = not self.training and suffixes is not None

        # Call encoder: Differentiate between static and dynamic attributes
        # and apply zero padding mask:
        (h_enc, c_enc) = self.encoder(input=prefixes, static_inputs=static_inputs, mask=prefix_mask)

        # Get SOS event: Last prefx event:
        cat_prefixes, num_prefixes = prefixes
        cat_sos_events = [cat_tens[:, -1:] for cat_tens in cat_prefixes]
        num_sos_events = [num_tens[:, -1:] for num_tens in num_prefixes]
        sos_event = [cat_sos_events, num_sos_events]

        # output_sizes is a list of two dicts: [cat_dict, num_dict]
        cat_output_features_labels, _ = self.output_sizes
        # Prediction dictionary for categorical features
        cat_predictions = {f"{key}_{suffix}": None
                           for key in cat_output_features_labels
                           for suffix in ["mean", "var"]}
        predictions = [cat_predictions, {}]

        # Training
        if training:
            batch_size = cat_prefixes[0].shape[0]
            tf_mask = torch.zeros(batch_size, self.seq_len_pred, device=cat_prefixes[0].device)
            # Timestep iterations: 0, 1, ..., n-1
            for t in range(self.seq_len_pred):
                # SOS Event
                if t == 0:
                    # Step 0 always consumes the last prefix event (ground truth).
                    tf_mask[:, t] = 1.0
                    # preds: list containing two dicts one for all means (cat, num),
                    # one for all vars (cat, num)
                    preds, (h, c), z = self.decoder(input=sos_event, hx=(h_enc, c_enc), z=None, pred=False)
                    pred_means, pred_vars = preds

                # Next Event
                # Decide per timestep whether to use teacher forcing
                else:
                    # Random value for teacher forcing for each timestep: If smaller use target else predicted. For high teacher forcing use target
                    teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                    if teacher_force:
                        tf_mask[:, t] = 1.0
                        # Use ground-truth previous event
                        cat_t_suffix_event = [cat_tens[:, t - 1 : t] for cat_tens in suffixes[0]]
                        num_t_suffix_event = [num_tens[:, t - 1 : t] for num_tens in suffixes[1]] 
                        
                        t_suffix_event = [cat_t_suffix_event, num_t_suffix_event]

                        preds, (h, c), _ = self.decoder(input=t_suffix_event, hx=(h, c), z=z, pred=False)
                        pred_means, pred_vars = preds

                    else:
                        # Use model prediction
                        last_pred_event = self.__transform_pred_into_next_event(pred_means=pred_means, pred_index=t, suffix=suffixes)
                        # For prediction, we assume valid input (or we could carry over
                        # mask if we wanted to propagate padding)
                        # But usually we don't mask predictions during generation unless
                        # we track finished state.
                        preds, (h, c), _ = self.decoder(input=last_pred_event, hx=(h, c), z=z, pred=True)
                        pred_means, pred_vars = preds

                cat_pred_means, _ = pred_means
                cat_pred_vars, _ = pred_vars

                # Add categorical tensors to output
                for key in cat_output_features_labels:
                    if t == 0:
                        predictions[0][f"{key}_mean"] = cat_pred_means[f"{key}_mean"].unsqueeze(0)
                        predictions[0][f"{key}_var"] = cat_pred_vars[f"{key}_var"].unsqueeze(0)
                    else:
                        predictions[0][f"{key}_mean"] = torch.cat((predictions[0][f"{key}_mean"], cat_pred_means[f"{key}_mean"].unsqueeze(0)),dim=0)
                        predictions[0][f"{key}_var"] = torch.cat((predictions[0][f"{key}_var"], cat_pred_vars[f"{key}_var"].unsqueeze(0)), dim=0)

        # Validation:
        if validation:
            for k in range(self.seq_len_pred):
                if k == 0:
                    preds, (h, c), z = self.decoder(input=sos_event, hx=(h_enc, c_enc), z=None, pred=False)
                    pred_means, pred_vars = preds
                else:
                    last_pred_event = self.__transform_pred_into_next_event(pred_means=pred_means)
                    preds, (h, c), z = self.decoder(input=last_pred_event, hx=(h, c), z=z, pred=True)
                    pred_means, pred_vars = preds

                cat_pred_means, _ = pred_means
                cat_pred_vars, _ = pred_vars

                # Add categorical tensors to output
                for key in cat_output_features_labels:
                    if k == 0:
                        predictions[0][f"{key}_mean"] = cat_pred_means[f"{key}_mean"].unsqueeze(0)
                        predictions[0][f"{key}_var"] = cat_pred_vars[f"{key}_var"].unsqueeze(0)
                    else:
                        predictions[0][f"{key}_mean"] = torch.cat((predictions[0][f"{key}_mean"],
                                                                   cat_pred_means[f"{key}_mean"].unsqueeze(0)),
                                                                 dim=0)
                        predictions[0][f"{key}_var"] = torch.cat((predictions[0][f"{key}_var"], cat_pred_vars[f"{key}_var"].unsqueeze(0)),
                                                                 dim=0)

        # Return training or validation output
        if training and return_teacher_forcing_mask:
            return predictions, (h, c), self.seq_len_pred, self.output_feature_indeces, tf_mask

        return predictions, (h, c), self.seq_len_pred, self.output_feature_indeces

    def __transform_pred_into_next_event(self,
                                         pred_means,
                                         pred_index: Optional[int] = None,
                                         suffix: Optional[list] = None):
        """
        Transform predictions into next event for decoder input.
        
        Inputs:
        - pred_means: predicted values
        - pred_index: index of event for next prediction
        - suffix: Target data

        Outputs:
        - next_event: event in decoder input data format
        """
        cat_pred_means, _ = pred_means

        # Create index tensor based on predicted logits
        cat_preds = [torch.argmax(tensor, dim=1).unsqueeze(1) for _, tensor in enumerate(cat_pred_means.values())]

        # Keep continuous decoder inputs even though they are not predicted as outputs.
        if suffix is not None and pred_index is not None and pred_index > 0:
            _, num_suffix = suffix
            num_events = [num_suffix[i][:, pred_index - 1 : pred_index] for i in self.data_indices_dec[1]]
        else:
            first_cat_tensor = next(iter(cat_pred_means.values()))
            batch_size = first_cat_tensor.shape[0]
            num_events = [torch.zeros(batch_size, 1, device=first_cat_tensor.device) for _ in self.data_indices_dec[1]]

        last_event = [cat_preds, num_events]

        return last_event

    # During test time:
    def inference(self,
                  # dynamic event attributes for encoder
                  prefix: Optional[list] = None,
                  # static event attributes for encoder
                  static_inputs: Optional[Union[Tensor, List, Tuple, dict]] = None,
                  mask: Optional[Tensor] = None,
                  # last prefix event (decoder (dynamic) event attributes only)
                  last_event: Optional[list] = None,
                  hx: Optional[Tuple[Tensor, Tensor]] = None,
                  z: Optional[Tuple[List, List]] = None):
        
        """
        Inference method fo scenario analysis based on Monte Carlo sampling.

        Inputs:
        - prefix: Input sequence of the model to be analyzed by encoder.
        - static_inputs: Optional static attribute tensor(s) to merge with the latent
        - mask: Zero padding mask for prefix.
        - last_event: Last event which was the output of the decoder.
        - hx: Last hidden state which was the output of the decoder.

        Outputs:
        - predictions: Predicted outcome:
            - [categorical dict (key: feature name, value tensor),
            - numerical dict (key: feature name, value tensor)]
            - (h,c): Predicted last hidden and cell state
        """
        with torch.no_grad():
            # First Prediciton
            if prefix is not None:
                # Call encoder (static inputs are only used here)
                (h_enc, c_enc) = self.encoder(input=prefix, static_inputs=static_inputs, mask=mask)

                # Get SOS event: Last prefx event:
                cat_prefixes, num_prefixes = prefix
                cat_sos_events = [cat_tens[:, -1:] for cat_tens in cat_prefixes]
                num_sos_events = [num_tens[:, -1:] for num_tens in num_prefixes]
                sos_event = [cat_sos_events, num_sos_events]

                preds, (h, c), z = self.decoder(input=sos_event, hx=(h_enc, c_enc), z=None, pred=False)

                # Return the sample masks for consistent variational inference
                return preds, (h, c), z

            # Second-n_th prediction
            else:
                (h, c) = hx
                preds, (h, c), _ = self.decoder(input=last_event, hx=(h, c), z=z, pred=True)
                return preds, (h, c)

    # save and load the trained models
    def save(self, path: str):
        """
        Store the trained model at path.
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "kwargs": {"data_set_categories": self.data_set_categories,
                       "enc_feat": self.enc_feat,
                       "dec_feat": self.dec_feat,
                       "seq_len_pred": self.seq_len_pred,
                       "hidden_size": self.hidden_size,
                       "num_layers": self.num_layers,
                       "dropout": self.dropout,
                       "static_data_set_categories": self.static_data_set_categories,
                       "static_enc_feat": self.static_enc_feat,},
            }
        return torch.save(checkpoint, path)

    @staticmethod
    def load(path: str, dropout: Optional[float] = None):
        """
        Load the stored model at path.
        """
        checkpoint = torch.load(path, weights_only=False, map_location=torch.device("cpu"))
        if dropout is not None:
            checkpoint["kwargs"]["dropout"] = dropout
        checkpoint["kwargs"].setdefault("static_data_set_categories", None)
        checkpoint["kwargs"].setdefault("static_enc_feat", None)
        checkpoint["kwargs"].pop("static_input_size_enc", None)
        model = DropoutUncertaintyEncoderDecoderLSTM(**checkpoint["kwargs"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


class DropoutUncertaintyLSTMDecoder(nn.Module):
    """
    Decoder part of the ED-LSTM with MC Dropout for uncertainty estimation.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_sizes: dict,
                 embeddings,
                 data_indices_dec,
                 num_layers: int,
                 dropout: float):
        """
        Decoder part of the Encoder-Decoder LSTM.

        Args:
        - input_size (int): Size of input event attributes
        - hidden_size (int): Size of hidden layers
        - output_sizes (dict): Tuple of dictionaries for categorical and numerical output feature sizes
        - embeddings: Categorical event attributes embeddings
        - data_indices_dec: Indices of event attributes
        - num_layers (int): Number of hidden layers in the LSTM
        - dropout (float): Dropout probability, must be in [0, 1). Required.
        """
        super(DropoutUncertaintyLSTMDecoder, self).__init__()

        self.embeddings = embeddings

        self.data_indices_dec = data_indices_dec

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.act = nn.ReLU()

        # Create a first cell:
        self.first_layer = DropoutUncertaintyLSTMCell(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout)

        # Create multiple LSTM cells based on num_layer
        self.hidden_layers = nn.ModuleList([DropoutUncertaintyLSTMCell(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout)for _ in range(num_layers - 1)])

        self.output_sizes = output_sizes

        # Output_sizes is a list containing two dicts:
        # one for categorical features and one (possibly empty) numerical features
        cat_output_sizes, _ = output_sizes

        # Create a ModuleDict to hold the layers
        self.output_layers = nn.ModuleDict()
        # Dynamically create mean and variance output linear layers
        # for categorical features
        for key, size in cat_output_sizes.items():
            self.output_layers[f"{key}_mean"] = nn.Linear(hidden_size, size)
            self.output_layers[f"{key}_var"] = nn.Linear(hidden_size, size)

    def regularizer(self):
        """
        L2 regularization of Encoder weights, biases and dropout.
        """
        total_weight_reg, total_bias_reg = self.first_layer.regularizer()

        for hidden_layer in self.hidden_layers:
            weight, bias = hidden_layer.regularizer()

            total_weight_reg += weight
            total_bias_reg += bias

        # Projection layer (weaker prior)
        proj_weight_reg = 0.1 * torch.sum(self.input_proj.weight**2)
        proj_bias_reg = 0.1 * torch.sum(self.input_proj.bias**2)
        total_weight_reg += proj_weight_reg
        total_bias_reg += proj_bias_reg

        return total_weight_reg, total_bias_reg

    def __data_enc_for_model(self, data, pred):
        """
        Transofrms prefix or suffix input into a tensor structure for the encoder.
        """
        if pred:
            cats, nums = data
            nums = nums if nums is not None else []
        else:
            # cat dims: list (n categorical values):
            # Each with Tensor: batch_size x (window_size - suffix size)
            cats = [data[0][i] for i in self.data_indices_dec[0]]
            # num dims: list (n numerical values):
            # Each with Tensor: batch_size x (window_size - suffix size)
            nums = [data[1][i] for i in self.data_indices_dec[1]]

        if len(nums) == 0 and len(self.data_indices_dec[1]) > 0:
            batch_size = cats[0].shape[0]
            seq_len = cats[0].shape[1]
            
            nums = [torch.zeros(batch_size, seq_len, device=cats[0].device) for _ in self.data_indices_dec[1]]

        assert len(cats) == len(self.data_indices_dec[0]) and len(nums) == len(self.data_indices_dec[1]
                                                                               ), "Decoder: Number of input tensor is unequal the number of indices"

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
        next_event = torch.cat((merged_cats, merged_nums), dim=-1).permute(1, 0, 2)  # dim: seq_len x batch_size x input_features
        return next_event

    def forward(self,
                input: Tensor,
                hx: Tuple[Tensor, Tensor],
                z: Optional[Tuple[List, List]] = None,
                pred: Optional[bool] = True) -> Tuple[list, Tuple[Tensor, Tensor]]:
        """
        Prediction of next event based on last hidden state and last event.

        Inputs:
        - input_event: Either last sequence event or next predicted, target event: Tensor: seq_len (1) x batch_size x input_features
        - hx: Tuple containing last hidden state and cell state of the encoder: Tensor: batch_size x hidden_size

        Outputs:
        - predictions: List containing:
        - activity mean,
        - activity log variance,
        - timestamp mean,
        - timestamp log variance.
        - h, c: Updated hidden and cell states.
        """
        prediction_means = [{}, {}] 
        prediction_vars = [{}, {}] 

        # Process the input event through the encoder
        event = self.__data_enc_for_model(data=input, pred=pred)  # dim: Tensor: seq_len x batch_size x input feature

        input_proj = self.input_proj(event)
        input_proj = self.layernorm(input_proj)
        input_proj = self.act(input_proj)

        # first decoder call initialize sample mask
        if z is None:
            z_hidden_layers = []
            # Pass input_event through the first LSTM layer and all hidden layers
            outputs, (h, c), z_first_layer = self.first_layer(input=input_proj, hx=hx, z=None)

            for _, lstm_cell in enumerate(self.hidden_layers):
                outputs, (h, c), z_hidden_layer = lstm_cell(input=outputs, hx=(h, c), z=None)
                z_hidden_layers.append(z_hidden_layer)

            z = (z_first_layer, z_hidden_layers)
        # Use same sample masks from previous iterations for decoder
        else:
            # Pass input_event through the first LSTM layer and all hidden layers
            outputs, (h, c), _ = self.first_layer(input=input_proj, hx=hx, z=z[0])

            for i, lstm_cell in enumerate(self.hidden_layers):
                outputs, (h, c), _ = lstm_cell(input=outputs, hx=(h, c), z=z[1][i])

        # Get the last output (outputs[-1]) for predictions
        final_output = outputs[-1]

        # Unpack output_sizes into categorical and numerical dicts
        cat_output_sizes, _ = self.output_sizes

        # Predict means and variances for categorical features
        for key in cat_output_sizes:
            pred_mean = self.output_layers[f"{key}_mean"](final_output)
            prediction_means[0][f"{key}_mean"] = (
                pred_mean  # Store in the first dict (for cat features)
            )

            pred_var = self.output_layers[f"{key}_var"](final_output)
            prediction_vars[0][f"{key}_var"] = (
                pred_var  # Store in the first dict (for cat features)
            )

        predictions = [prediction_means, prediction_vars]

        # Return the prediction dictionaries for means and variances
        # along with the hidden states
        return predictions, (h, c), z


class DropoutUncertaintyLSTMEncoder(nn.Module):
    """
    Encoder part of the Encoder-Decoder LSTM with MC-Dropout uncertainty estimation.
    """

    def __init__(self,
                 hidden_size: int,
                 num_layers: int,
                 # dynamic attributes
                 embeddings,
                 data_indices_enc: list,
                 input_size: int,
                 # static attributes
                 static_embeddings: Optional[nn.ModuleList] = None,
                 static_data_indices: Optional[List[List[int]]] = None,
                 static_input_size: Optional[int] = 0,
                 # mc-dropout
                 dropout: float = 0.0):
        """
        Encoder part of the Encoder-Decoder LSTM.

        Args:
        - hidden_size (int): Size of the LSTM hidden state.
        - num_layers (int): Number of stacked LSTM layers.
        - embeddings (nn.ModuleList): Embedding modules for dynamic categorical encoder inputs.
        - data_indices_enc (list): Indices selecting dynamic categorical and numerical tensors for the encoder.
        - input_size (int): Number of dynamic input features per timestep.
        - static_embeddings (Optional[nn.ModuleList]): Embedding modules for static categorical inputs.
        - static_data_indices (Optional[List[List[int]]]): Indices selecting static categorical and numerical tensors.
        - static_input_size (Optional[int]): Flattened size of all static features after embeddings.
        - dropout (float): Dropout probability, must be in [0, 1). Required.
        """
        super(DropoutUncertaintyLSTMEncoder, self).__init__()

        # Embeddings for dynamic
        self.embeddings = embeddings
        # for static
        if static_embeddings is not None:
            self.static_embeddings = static_embeddings

        # List of two lists (categorical, numerical)
        # each containing the indices of tensors required for encoder
        self.data_indices_enc = data_indices_enc
        # for static
        if static_data_indices is not None:
            self.static_data_indices = static_data_indices
        # Static features are concatenated
        # to the dynamic per-timestep features and fed into the LSTM.
        self.static_input_size = static_input_size or 0

        # Linear projection before inserted into LSTM layers
        total_input_size = input_size + (self.static_input_size if self.static_input_size > 0 else 0)
        self.input_proj = nn.Linear(total_input_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.act = nn.ReLU()

        # Create a first cell:
        self.first_layer = DropoutUncertaintyLSTMCell(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout)

        # Create multiple LSTM cells based on num_layer
        self.hidden_layers = nn.ModuleList([DropoutUncertaintyLSTMCell(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout)
                                            for _ in range(num_layers - 1)])

    def regularizer(self) -> Tuple[float, float]:
        """
        L2 regularization of Encoder weights, biases and dropout.
        """
        
        total_weight_reg, total_bias_reg = self.first_layer.regularizer()

        for layer in self.hidden_layers:
            weight, bias = layer.regularizer()
            total_weight_reg += weight
            total_bias_reg += bias

        # Projection layer (weaker prior)
        proj_weight_reg = 0.1 * torch.sum(self.input_proj.weight**2)
        proj_bias_reg = 0.1 * torch.sum(self.input_proj.bias**2)
        total_weight_reg += proj_weight_reg
        total_bias_reg += proj_bias_reg

        return total_weight_reg, total_bias_reg

    def forward(self,
                input: List,
                static_inputs: Optional[Union[Tensor, List, Tuple, dict]] = None,
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the encoder.

        Inputs:
        - input: Prefixes, Tensor: seq_len, batch_size, input_size
        - static_inputs: inputs that are static for the whole case
        - mask: zero padd mask

        Output:
        - h,c: Last hidden and cell states of the last layer.
        """
        # Transform the input into a single tensor: [T, B, dyn_features]
        prefixes = self.__data_enc_model(
            data=input
        )  # dim: seq_len x batch_size x input_features

        # Optionally concatenate static features across the time axis T.
        if self.static_input_size > 0:
            static_tensor = self.__static_data_enc_model(
                static_inputs, device=prefixes.device, dtype=prefixes.dtype
            )
            if static_tensor is not None:
                # Expand to [T, B, static_features] and concat
                static_seq = static_tensor.unsqueeze(0).expand(
                    prefixes.shape[0], -1, -1
                )
                prefixes = torch.cat((prefixes, static_seq), dim=-1)

        # Project input features to hidden size
        prefixes = self.input_proj(prefixes)
        prefixes = self.layernorm(prefixes)
        prefixes = self.act(prefixes)

        # zero masking
        mask_seq = None
        if mask is not None:
            seq_len = prefixes.shape[0]
            if mask.shape[1] != seq_len:
                # Assuming left-aligned prefix (standard for [:-suffix]),  sk
                mask = mask[:, :seq_len]
            mask_seq = (mask.to(device=prefixes.device, dtype=prefixes.dtype).transpose(0, 1).contiguous())

            # Apply mask to prefixes to ensure padded inputs are zero
            prefixes = prefixes * mask_seq.unsqueeze(-1)

        outputs, (h, c), _ = self.first_layer(input=prefixes, hx=None, z=None, mask=mask_seq)

        # Pass through the remaining LSTM cell: Layer gets for: input: h_n Tensor,
        # hx: (h, c)
        for _, layer in enumerate(self.hidden_layers):
            outputs, (h, c), _ = layer(input=outputs, hx=(h, c), z=None, mask=mask_seq)

        return (h, c)

    def __data_enc_model(self, data):
        """
        Dynamic attribute model encoder.
        """
        # cats dims: list (n categorical values):
        # Each with Tensor: batch_size x (window_size - suffix size)
        cats = [data[0][i] for i in self.data_indices_enc[0]]
        # nums dims: list (n numerical values):
        # Each with Tensor: batch_size x (window_size - suffix size)
        nums = [data[1][i] for i in self.data_indices_enc[1]]

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
        prefixes = torch.cat((merged_cats, merged_nums), dim=-1).permute(1, 0, 2)  # dim: seq_len x batch_size x input_features
        return prefixes

    def __static_data_enc_model(self,
                                static_inputs: Optional[Union[Tensor, List, Tuple, dict]],
                                device: Optional[torch.device] = None,
                                dtype: Optional[torch.dtype] = None,) -> Optional[Tensor]:
        """
        Static attribute model encoder.
        """
        if static_inputs is None or self.static_input_size == 0:
            return None

        # Allow passing either a (static_cats, static_nums) tuple or a dict.
        static_cats = None
        static_nums = None
        if isinstance(static_inputs, dict):
            static_cats = static_inputs.get("static_cats", static_inputs.get("cats", None))
            
            static_nums = static_inputs.get("static_nums", static_inputs.get("nums", None))
        elif isinstance(static_inputs, (list, tuple)):
            if len(static_inputs) != 2:
                raise TypeError("static_inputs tuple/list must be (static_cats, static_nums)")
            static_cats, static_nums = static_inputs
        else:
            raise TypeError("static_inputs must be a tuple/list(static_cats, static_nums) or a dict")

        merged_static_cats = None
        if static_cats is not None and len(self.static_embeddings) > 0:
            # Support either a single tensor [B, n_static_cats]
            # or a list of tensors [B]
            if isinstance(static_cats, Tensor):
                static_cats = static_cats.long()
                # If 1D input [n_features] (inference single case),
                # add batch dim -> [1, n_features]
                if static_cats.dim() == 1:
                    static_cats = static_cats.unsqueeze(0)
            else:
                # List of tensors case
                static_cats = torch.stack([t.long() for t in static_cats], dim=1)

            embedded = []
            for i, emb in enumerate(self.static_embeddings):
                embedded.append(emb(static_cats[:, i]))
            merged_static_cats = torch.cat(embedded, dim=-1)

        merged_static_nums = None
        if static_nums is not None:
            if isinstance(static_nums, Tensor):
                # bring to size (features x B)
                if static_nums.dim() == 1:
                    static_nums = static_nums.unsqueeze(0)
                merged_static_nums = static_nums
            else:
                merged_static_nums = torch.cat([num.unsqueeze(1) for num in static_nums], dim=-1)

        if merged_static_cats is not None and device is not None:
            merged_static_cats = merged_static_cats.to(device=device, dtype=dtype)
        if merged_static_nums is not None and device is not None:
            merged_static_nums = merged_static_nums.to(device=device, dtype=dtype)

        if merged_static_cats is not None and merged_static_nums is not None:
            return torch.cat((merged_static_cats, merged_static_nums), dim=-1)
        elif merged_static_cats is not None:
            return merged_static_cats
        elif merged_static_nums is not None:
            return merged_static_nums
        else:
            return None


class DropoutUncertaintyLSTMCell(nn.Module):
    """
    LSTM cell with MC Dropout for uncertainty estimation.
    """
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        """
        Initializes LSTM cell with MC Dropout.

        Args:
        - input_size (int): Size of input features.
        - hidden_size (int): Size of hidden layer.
        - dropout (float): Dropout probability, must be in [0, 1).
        """
        super(DropoutUncertaintyLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Validate and set fixed dropout probability
        if not isinstance(dropout, (int, float)):
            raise TypeError("Dropout rate must be a float, got: " + str(type(dropout)))
        if not 0 <= dropout < 1:
            raise ValueError("Dropout rate must be in [0, 1), got: " + str(dropout))
        self.p_logit = float(dropout)

        # Input gate
        self.Wi = nn.Linear(self.input_size, self.hidden_size)
        self.Ui = nn.Linear(self.hidden_size, self.hidden_size)
        # Forget gate
        self.Wf = nn.Linear(self.input_size, self.hidden_size)
        self.Uf = nn.Linear(self.hidden_size, self.hidden_size)
        # Cell state gate
        self.Wc = nn.Linear(self.input_size, self.hidden_size)
        self.Uc = nn.Linear(self.hidden_size, self.hidden_size)
        # Output gate
        self.Wo = nn.Linear(self.input_size, self.hidden_size)
        self.Uo = nn.Linear(self.hidden_size, self.hidden_size)

        self.init_weights()

    def init_weights(self):
        """
        Initializes weight layers with initial values.
        """
        k = torch.tensor(self.hidden_size, dtype=torch.float32).reciprocal().sqrt()

        # Input gate weights:
        self.Wi.weight.data.uniform_(-k, k)
        self.Wi.bias.data.uniform_(-k, k)
        self.Ui.weight.data.uniform_(-k, k)
        self.Ui.bias.data.uniform_(-k, k)

        # Forget gate weights
        self.Wf.weight.data.uniform_(-k, k)
        self.Wf.bias.data.uniform_(-k, k)
        self.Uf.weight.data.uniform_(-k, k)
        self.Uf.bias.data.uniform_(-k, k)

        # Cell state gate weights
        self.Wc.weight.data.uniform_(-k, k)
        self.Wc.bias.data.uniform_(-k, k)
        self.Uc.weight.data.uniform_(-k, k)
        self.Uc.bias.data.uniform_(-k, k)

        # Output gate weights
        self.Wo.weight.data.uniform_(-k, k)
        self.Wo.bias.data.uniform_(-k, k)
        self.Uo.weight.data.uniform_(-k, k)
        self.Uo.bias.data.uniform_(-k, k)

    def _mc_dropout_sample_mask(self, B: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """
        Applies dropout to the LSTM Cell weight layers.

        INPUTS:
        B: Batch size

        OUTPUTS:
        zx: Dropout mask for weight layer before input
        zh: Dropout mask for weight layer before hidden

        Note: value p_logit at infinity can cause numerical instability.
        Dropout masks for 4 gates, scale input by 1 / (1 - p)
        """
        p = self.p_logit

        # Four Weight matrix pairs: Perform dropout for each weight layer.
        GATES = 4

        eps = torch.tensor(1e-7, device=device, dtype=torch.float32)
        t = 1e-1

        # tensors with random values:
        ux = torch.rand(GATES, B, self.input_size, device=device, dtype=torch.float32) # dim gates x batch_size x input_size
        uh = torch.rand(GATES, B, self.hidden_size, device=device, dtype=torch.float32) # dim (gates=weight matrices per cell x batch_size x hidden_size)

        # Dropout masks: containing values near 1 for keeping weights,
        # and near 0 for dropping weights for each gate and batch
        if self.input_size == 1:
            zx = 1 - torch.sigmoid((torch.log(eps)- torch.log(1 + eps)+ torch.log(ux + eps)- torch.log(1 - ux + eps))/ t)
        else:
            # dim: gates x batch_size x input_features
            zx = (1- torch.sigmoid((torch.log(p + eps)- torch.log(1 - p + eps)+ torch.log(ux + eps)- torch.log(1 - ux + eps))/ t)) / (1 - p)
        # dim: gates x batch_size x input_features
        zh = (1- torch.sigmoid((torch.log(p + eps)- torch.log(1 - p + eps)+ torch.log(uh + eps)- torch.log(1 - uh + eps))/ t)) / (1 - p)

        return zx, zh

    def regularizer(self):
        """
        L2 regularization of weights and biases scaled for dropout.
        """
        p = self.p_logit

        # Weight L2 sum (keeps autograd).
        # For MC-dropout-as-variational-inference:
        # the KL/L2 term is typically scaled by (1-p) rather than 1/(1-p).
        keep_prob = 1.0 - p
        weight_sum = ( sum(torch.sum(params**2) for name, params in self.named_parameters() if name.endswith("weight")) * keep_prob)

        # Bias L2 sum
        bias_sum = sum(torch.sum(params**2) for name, params in self.named_parameters() if name.endswith("bias"))

        return weight_sum, bias_sum

    def forward(self,
                input: Tensor,
                hx: Optional[Tuple[Tensor, Tensor]] = None,
                z: Optional[Tuple[Tensor, Tensor]] = None,
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        """
        Performs forward pass of LSTM cell with MC Dropout.

        Inputs:
        - input: Input tensor with shape (sequence, batch, input dimension)
        - hx: h_t: hidden state and c_t: cell state as tuple at time step (event t)
        - z: dropout masks for LSTM weights
        - mask:

        Outputs:
        - hn: List of all hidden states: h_1, ... h_n
        - (h_t, c_t): Last hidden and cell state
        - (zx, zh): Applied MC dropout masks
        """
        device = input.device
        T, B, _ = input.shape

        # Initialize hidden and cell states
        if hx is None:
            h_t = torch.zeros(B, self.hidden_size, device=device, dtype=input.dtype)
            c_t = torch.zeros(B, self.hidden_size, device=device, dtype=input.dtype)
        else:
            h_t, c_t = hx
            h_t = h_t.to(device=device, dtype=input.dtype)
            c_t = c_t.to(device=device, dtype=input.dtype)

        # Prepare mask: [T, B, 1]
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            mask = mask.to(device=device, dtype=input.dtype)

        # MC dropout masks
        if z is None:
            zx, zh = self._mc_dropout_sample_mask(B, device=device)
            # Ensure same device as input
            zx = [m.to(device=device, dtype=input.dtype) for m in zx]
            zh = [m.to(device=device, dtype=input.dtype) for m in zh]
        else:
            zx, zh = z
            zx = [m.to(device=device, dtype=input.dtype) for m in zx]
            zh = [m.to(device=device, dtype=input.dtype) for m in zh]

        # Prepare output storage
        hn = torch.empty(T, B, self.hidden_size, device=device, dtype=input.dtype)

        for t in range(T):
            x = input[t]  # [B, input_size]

            # Apply MC dropout per gate explicitly
            x_i = x * zx[0]
            x_f = x * zx[1]
            x_c = x * zx[2]
            x_o = x * zx[3]

            h_i = h_t * zh[0]
            h_f = h_t * zh[1]
            h_c = h_t * zh[2]
            h_o = h_t * zh[3]

            # LSTM gates
            i = torch.sigmoid(self.Wi(x_i) + self.Ui(h_i))
            f = torch.sigmoid(self.Wf(x_f) + self.Uf(h_f))
            g = torch.tanh(self.Wc(x_c) + self.Uc(h_c))
            o = torch.sigmoid(self.Wo(x_o) + self.Uo(h_o))

            # Update cell and hidden
            c_new = f * c_t + i * g
            h_new = o * torch.tanh(c_new)

            # Apply prefix mask: keep old states where input is padding
            if mask is not None:
                step_mask = mask[t]  # [B, 1]
                c_t = step_mask * c_new + (1 - step_mask) * c_t
                h_t = step_mask * h_new + (1 - step_mask) * h_t
            else:
                c_t, h_t = c_new, h_new

            hn[t] = h_t

        return hn, (h_t, c_t), (zx, zh)