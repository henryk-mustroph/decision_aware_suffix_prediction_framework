import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    LSTM prefix encoder
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.hid_dim = hidden_size
        self.n_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=False)

    def forward(self, x):
        self.lstm.flatten_parameters()
        output, (h, c) = self.lstm(x)
        return h, c


class Decoder(nn.Module):
    """
    LSTM suffix decoder with fc_out + ReLU.

    Also exposes hidden_state (B, T, H) so auxiliary heads can predict
    non-activity dynamic event attributes for autoregressive decoding.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.hid_dim = hidden_size
        self.n_layers = num_layers
        self.output_dim = input_size

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

        self.fc_out = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()

    def forward(self, input, hidden, cell, return_hidden_seq: bool = False):
        self.rnn.flatten_parameters()
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        prediction = self.relu(self.fc_out(output))
        if return_hidden_seq:
            return prediction, hidden, cell, output
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    """
    Sequence-to-sequence generator combining Encoder and Decoder
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, start_input, teacher_forcing_ratio=0.5,
                return_teacher_forcing_mask: bool = False,
                return_hidden_seq: bool = False):
        hidden, cell = self.encoder(src)

        predictions = []
        hidden_seq = []
        inp = start_input
        tf_mask = torch.zeros(trg.size(0), trg.size(1), device=src.device)

        for i in range(trg.size(1)):
            # Keep semantics aligned with UED:
            # tf_mask[:, i] == 1 iff decoder step i consumed a ground-truth input token.
            # Step 0 always consumes the last prefix event (start_input), i.e. ground truth.
            if i == 0:
                tf_mask[:, i] = 1.0
            else:
                teacher_force = random.random() < teacher_forcing_ratio
                if teacher_force:
                    tf_mask[:, i] = 1.0
                    # For decoder step i, consume previous true suffix event y_{i-1}.
                    inp = trg[:, i - 1 : i, :]
                else:
                    # For decoder step i, consume previous model prediction y_hat_{i-1}.
                    inp = output

            output, hidden, cell, hidden_step = self.decoder(inp, hidden, cell, return_hidden_seq=True)
            predictions.append(output)
            hidden_seq.append(hidden_step)

        prediction = torch.cat(predictions, dim=1)
        hidden_seq_tensor = torch.cat(hidden_seq, dim=1) if return_hidden_seq else None

        if return_teacher_forcing_mask and return_hidden_seq:
            return prediction, tf_mask, hidden_seq_tensor
        if return_teacher_forcing_mask:
            return prediction, tf_mask
        if return_hidden_seq:
            return prediction, hidden_seq_tensor
        return prediction


class Discriminator(nn.Module):
    """
    Suffix sequence discriminator
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super().__init__()
        self.hid_dim = hidden_size
        self.n_layers = num_layers

        self.rnn = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, input):
        self.rnn.flatten_parameters()
        output, (hidden, cell) = self.rnn(input)
        prediction = self.fc_out(output)
        return prediction


def init_weights(m):
    """
    Normal weight initialization standard normal distribution.
    """
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean=0.0, std=0.08)



# Wrapper that composes the reference modules with an embedding layer and provides the interface expected by the training / inference pipeline.
class TaymouriAdversarialLSTM(nn.Module):
    """
    GAN-LSTM for suffix prediction.

    Wraps the reference Encoder, Decoder, Seq2Seq and Discriminator
    with an embedding layer for mixed categorical + numerical event attributes
    """

    def __init__(self,
                 data_set_categories: list[tuple[str, dict[str, int]]],
                 model_feat: list,
                 concept_name_id: int,
                 hidden_size: int,
                 num_layers: int,
                 seq_len_pred: int,
                 input_size: int = 1,
                 output_size_act: int | None = None,
                 dropout: float = 0.2):

        super().__init__()

        self.data_set_categories = data_set_categories
        self.model_feat = model_feat
        self.concept_name_id = concept_name_id
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len_pred = seq_len_pred
        self.dropout = dropout

        cat_categories, num_categories = data_set_categories
        cat_input_feat_model, num_input_feat_model = model_feat
        cat_dict = {cat[0]: cat[1] for cat in cat_categories}
        self.activity_feature_name = cat_categories[concept_name_id][0]
        if self.activity_feature_name not in cat_input_feat_model:
            raise ValueError(f"Activity feature '{self.activity_feature_name}' must be part of model_feat categorical inputs.")

        self.activity_input_index = cat_input_feat_model.index(self.activity_feature_name)

        classes_per_cat = [cat_dict[feat] for feat in cat_input_feat_model if feat in cat_dict]
        if len(classes_per_cat) == 0:
            raise ValueError("At least one categorical input feature is required.")

        # Track non-activity features so we can attach auxiliary prediction heads.
        self._other_cat_feature_names = [feat for feat in cat_input_feat_model
                                         if feat in cat_dict and feat != self.activity_feature_name]
        self._other_cat_class_counts = {feat: cat_dict[feat] for feat in self._other_cat_feature_names}
        self._num_feature_names = list(num_input_feat_model)

        # Embeddings
        self.embeddings = nn.ModuleList([nn.Embedding(n_cat, min(600, round(1.6 * n_cat**0.56))) for n_cat in classes_per_cat])

        embedding_size = sum(emb.embedding_dim for emb in self.embeddings)

        if input_size == 1:
            self.input_size = embedding_size + len(num_input_feat_model)
        else:
            self.input_size = input_size

        if output_size_act is None:
            output_size_act = classes_per_cat[concept_name_id]
        self.output_size_act = output_size_act

        # Seq2Seq (Encoder + Decoder)
        encoder = Encoder(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=dropout)
        
        decoder = Decoder(input_size=self.output_size_act,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=dropout)
        
        self.seq2seq = Seq2Seq(encoder, decoder)

        # Discriminator
        self.discriminator = Discriminator(input_size=self.output_size_act,
                                           hidden_size=self.hidden_size,
                                           num_layers=self.num_layers,
                                           dropout=dropout)

        # Auxiliary heads operating on the decoder hidden state. The single
        # autoregressive decoder LSTM is the shared recurrence; each output then
        # has its OWN dense layer over the decoder hidden sequence (the activity
        # itself is produced by the decoder's fc_out). These predict the
        # non-activity dynamic event attributes so the inference wrapper can roll
        # the prefix forward at decode time without GT leakage.
        self.other_cat_heads = nn.ModuleDict()
        for feat_name, n_classes in self._other_cat_class_counts.items():
            self.other_cat_heads[feat_name] = nn.Linear(self.hidden_size, n_classes)

        self.num_heads = nn.ModuleDict()
        for feat_name in self._num_feature_names:
            self.num_heads[feat_name] = nn.Linear(self.hidden_size, 1)

    # -backward-compatible accessors for the trainer 
    @property
    def discriminator_lstm(self):
        return self.discriminator.rnn

    @property
    def discriminator_head(self):
        return self.discriminator.fc_out

    # prefix handling (same pattern as FS_LSTM)
    def _build_prefix_tensor(self, prefixes):
        """
        Embed categorical + numerical prefix features into a single tensor.

        Input:
        - prefixes: [cats_list, nums_list] where each is a list of tensors of shape [B, T] (categorical ids) or [B, T] (numerical).
        
        Output:
        - Tensor of shape [B, T, input_size].
        """
        cats, nums = prefixes

        embedded_cats = [emb(cats[i]) for i, emb in enumerate(self.embeddings)]
        merged_cats = torch.cat(embedded_cats, dim=-1)

        if len(nums):
            merged_nums = torch.cat([num.unsqueeze(2) for num in nums], dim=-1)
        else:
            merged_nums = torch.tensor([], device=merged_cats.device)

        return torch.cat((merged_cats, merged_nums), dim=-1)

    def _build_start_input(self, prefixes):
        """
        Build the first decoder input from the last prefix event.
        """
        prefix_cats, _ = prefixes
        activity_ids = prefix_cats[self.activity_input_index][:, -1].long()
        return F.one_hot(activity_ids, self.output_size_act).float().unsqueeze(1)

    # generate suffixes
    def forward(self, prefixes, target_suffix=None, teacher_forcing_ratio: float = 0.0,
                return_teacher_forcing_mask: bool = False,
                return_aux_predictions: bool = False):
        """
        Inputs:
        - prefixes: [cats_list, nums_list] — prefix event features.
        - target_suffix: LongTensor [B, S] of activity ids (for teacher forcing) or None (free-running generation).
        - teacher_forcing_ratio: probability of feeding ground-truth token.
        - return_aux_predictions: when True, also returns dicts of non-activity attribute predictions.

        Outputs:
        - Activity logits with shape [S, B, output_size_act].
        - If return_aux_predictions: tuple with (logits, other_cat_logits dict, num_means dict)
          where each aux value has shape [S, B, C] or [S, B] respectively.
        """
        src = self._build_prefix_tensor(prefixes)
        start_input = self._build_start_input(prefixes)
        batch_size = src.size(0)
        max_len = self.seq_len_pred

        if target_suffix is not None:
            max_len = target_suffix.shape[1]
            trg = F.one_hot(target_suffix.long(), self.output_size_act).float()
        else:
            trg = torch.zeros(batch_size, max_len, self.output_size_act, device=src.device)

        seq2seq_output = self.seq2seq(src,
                                      trg,
                                      start_input,
                                      teacher_forcing_ratio,
                                      return_teacher_forcing_mask=return_teacher_forcing_mask,
                                      return_hidden_seq=return_aux_predictions)

        # Unpack possible return tuples.
        prediction = None
        tf_mask = None
        hidden_seq = None
        if return_teacher_forcing_mask and return_aux_predictions:
            prediction, tf_mask, hidden_seq = seq2seq_output
        elif return_teacher_forcing_mask:
            prediction, tf_mask = seq2seq_output
        elif return_aux_predictions:
            prediction, hidden_seq = seq2seq_output
        else:
            prediction = seq2seq_output

        activity_logits = prediction.permute(1, 0, 2)  # [S, B, C]

        if return_aux_predictions:
            # hidden_seq: [B, S, H] -> [S, B, H]. Each output has its own dense
            # head over the shared decoder hidden sequence.
            hs = hidden_seq.permute(1, 0, 2)
            other_cat_logits = {feat: head(hs) for feat, head in self.other_cat_heads.items()}
            num_means = {feat: head(hs).squeeze(-1) for feat, head in self.num_heads.items()}
            if return_teacher_forcing_mask:
                return activity_logits, tf_mask, other_cat_logits, num_means
            return activity_logits, other_cat_logits, num_means

        if return_teacher_forcing_mask:
            return activity_logits, tf_mask

        return activity_logits

    def discriminate(self, prefixes, suffix_activities):
        """
        Run discriminator on suffix activity sequences.

        Inputs:
        - prefixes: prefix tuple (kept for API compatibility).
        - suffix_activities

        Outputs:
        - Tensor [B] with probabilities of being real.
        """
        if suffix_activities.dtype == torch.long:
            suffix_input = F.one_hot(suffix_activities, self.output_size_act).float()
        else:
            suffix_input = suffix_activities

        prediction = self.discriminator(suffix_input)  # [B, S, 1]
        return torch.sigmoid(prediction[:, -1, 0])

    def sample_activity_ids(self, prefixes, max_len: int | None = None):
        """
        Greedy argmax decoding of activity ids.
        """
        logits = self.forward(prefixes=prefixes, target_suffix=None, teacher_forcing_ratio=0.0)
        if max_len is not None and logits.shape[0] != max_len:
            logits = logits[:max_len]
        return torch.argmax(logits, dim=-1).transpose(0, 1).contiguous()

    def beam_search(self, prefixes, beam_width: int = 3, max_len: int | None = None, eos_id: int | None = None,
                    return_aux_predictions: bool = False):
        """
        Beam-search decoding for activity suffixes.
        Returns all beam candidates as LongTensor [B, beam_width, max_len].

        When ``return_aux_predictions`` is True, also returns per-beam dicts of
        predicted non-activity attribute values per step (used by the inference
        wrappers to roll the prefix forward without GT leakage).
        """
        src = self._build_prefix_tensor(prefixes)
        h0, c0 = self.seq2seq.encoder(src)
        if max_len is None:
            max_len = self.seq_len_pred

        batch_size = src.size(0)
        device = src.device
        C = self.output_size_act
        predictions = []
        aux_predictions = []  # list per batch item; each: list per beam of dicts

        for b in range(batch_size):
            h_b = h0[:, b : b + 1, :].contiguous()
            c_b = c0[:, b : b + 1, :].contiguous()

            start_inp = self._build_start_input(([cat[b : b + 1] for cat in prefixes[0]], [num[b : b + 1] for num in prefixes[1]]))

            # Each beam state: (seq, score, prev_inp, h, c, done, aux_history)
            # aux_history is a list of dicts mapping feature_name -> sampled value (id for cat, float for num).
            beams = [([], 0.0, start_inp, h_b, c_b, False, [])]

            for _ in range(max_len):
                candidates = []
                for seq, score, prev_inp, h_prev, c_prev, done, aux_hist in beams:
                    if done:
                        candidates.append((seq, score, prev_inp, h_prev, c_prev, done, aux_hist))
                        continue

                    output, h_new, c_new, hidden_step = self.seq2seq.decoder(prev_inp, h_prev, c_prev, return_hidden_seq=True)
                    log_probs = F.log_softmax(output.squeeze(0).squeeze(0), dim=-1)

                    # Auxiliary predictions from the decoder hidden state for this step.
                    # hidden_step has shape [1, 1, hidden_size]; squeeze to [hidden_size].
                    hs = hidden_step.squeeze(0).squeeze(0)
                    step_other_cat_ids = {feat: int(torch.argmax(head(hs), dim=-1).item())
                                          for feat, head in self.other_cat_heads.items()}
                    step_num_means = {feat: float(head(hs).squeeze(-1).item())
                                      for feat, head in self.num_heads.items()}
                    step_aux = {"cat_ids": step_other_cat_ids, "num_values": step_num_means}

                    top_logp, top_idx = torch.topk(log_probs, k=min(beam_width, log_probs.shape[-1]))
                    for j in range(top_idx.shape[0]):
                        tok = int(top_idx[j].item())
                        tok_score = float(top_logp[j].item())
                        new_seq = seq + [tok]
                        is_done = eos_id is not None and tok == eos_id
                        # Feed one-hot of predicted token as next decoder input
                        next_inp = torch.zeros(1, 1, C, device=device)
                        next_inp[0, 0, tok] = 1.0
                        candidates.append((new_seq,
                                            score + tok_score,
                                            next_inp,
                                            h_new.clone(),
                                            c_new.clone(),
                                            is_done,
                                            aux_hist + [step_aux]))

                candidates.sort(key=lambda x: x[1], reverse=True)
                beams = candidates[:beam_width]

            # Return all beam candidates (padded to max_len)
            batch_beams = []
            batch_aux = []
            for beam_seq, _, _, _, _, _, aux_hist in beams:
                if len(beam_seq) < max_len:
                    beam_seq = beam_seq + [eos_id if eos_id is not None else 0] * (max_len - len(beam_seq))
                batch_beams.append(torch.tensor(beam_seq[:max_len], device=device, dtype=torch.long))
                batch_aux.append(aux_hist)
            predictions.append(torch.stack(batch_beams, dim=0))
            aux_predictions.append(batch_aux)

        token_tensor = torch.stack(predictions, dim=0)
        if return_aux_predictions:
            return token_tensor, aux_predictions
        return token_tensor

    def predict_next_event(self, input):
        """
        Single-step next-event prediction for decision-rule-guided decoding,
        mirroring FullShared_Join_LSTM.predict_next_event so the guided decoder
        can roll the prefix forward with the model's PREDICTED non-activity
        attributes (never ground truth). This is what feeds the decision model
        predicted dynamic event attributes during guided (beam) decode.

        `input` is the prefix already projected to this model's input features
        ([cats_list, nums_list], each [B, T]). The generator runs free-running and
        we read the activity + auxiliary attribute heads at the first predicted
        step.

        Output keys (matching FS):
          - "activity_id": Long tensor [B]
          - "cat_ids":    dict {feature_name: Long tensor [B]} (non-activity cats)
          - "num_values": dict {feature_name: Float tensor [B]} (numericals)
        """
        was_training = self.training
        if was_training:
            self.eval()
        try:
            with torch.no_grad():
                activity_logits, other_cat_logits, num_means = self.forward(
                    prefixes=input, target_suffix=None, teacher_forcing_ratio=0.0,
                    return_aux_predictions=True)
        finally:
            if was_training:
                self.train()

        # Aux heads predict every non-activity dynamic attribute in the input, so
        # the generator (encoder-decoder) predicts the same dynamic attributes it
        # reads. Take the first predicted step: [S, B, *] -> [B, *].
        activity_ids = torch.argmax(activity_logits[0], dim=-1)
        cat_ids = {feat: torch.argmax(logits[0], dim=-1) for feat, logits in other_cat_logits.items()}
        num_values = {feat: means[0].detach() for feat, means in num_means.items()}
        return {"activity_id": activity_ids,
                "cat_ids": cat_ids,
                "num_values": num_values}

    def init_weights_normal(self):
        """Initialize G and D parameters (Algorithm 1, step 1: standard normal distribution)."""
        self.apply(init_weights)


    def save(self, path: str):
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "kwargs": {"data_set_categories": self.data_set_categories,
                       "model_feat": self.model_feat,
                       "concept_name_id": self.concept_name_id,
                       "hidden_size": self.hidden_size,
                       "num_layers": self.num_layers,
                       "seq_len_pred": self.seq_len_pred,
                       "input_size": self.input_size,
                       "output_size_act": self.output_size_act,
                       "dropout": self.dropout},
            }
        return torch.save(checkpoint, path)

    @staticmethod
    def load(path: str):
        checkpoint = torch.load(path, weights_only=False, map_location=torch.device("cpu"))
        model = TaymouriAdversarialLSTM(**checkpoint["kwargs"])
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

