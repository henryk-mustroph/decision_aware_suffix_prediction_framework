import torch
import torch.nn as nn
import torch.nn.functional as F


class TaymouriAdversarialLSTM(nn.Module):
	"""
	GAN-style suffix model inspired by Taymouri et al. (2021).

	- Encoder consumes mixed categorical + numerical prefix attributes.
	- Generator predicts only activity logits for suffix steps.
	- Discriminator scores real/fake activity suffixes conditioned on prefix context.
	- Beam search decoding is provided for suffix generation.
	"""

	def __init__(
		self,
		data_set_categories: list[tuple[str, dict[str, int]]],
		model_feat: list,
		concept_name_id: int,
		hidden_size: int,
		num_layers: int,
		seq_len_pred: int,
		input_size: int = 1,
		output_size_act: int | None = None,
		dropout: float = 0.2,
	):
		super().__init__()

		self.data_set_categories = data_set_categories
		self.model_feat = model_feat
		self.concept_name_id = concept_name_id
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.seq_len_pred = seq_len_pred
		self.dropout = dropout

		cat_categories, _ = data_set_categories
		cat_input_feat_model, num_input_feat_model = model_feat
		cat_dict = {cat[0]: cat[1] for cat in cat_categories}

		classes_per_cat = [cat_dict[feat] for feat in cat_input_feat_model if feat in cat_dict]
		if len(classes_per_cat) == 0:
			raise ValueError("At least one categorical input feature is required.")

		self.embeddings = nn.ModuleList(
			[nn.Embedding(n_cat, min(600, round(1.6 * n_cat**0.56))) for n_cat in classes_per_cat]
		)
		embedding_size = sum([emb.embedding_dim for emb in self.embeddings])

		if input_size == 1:
			self.input_size = embedding_size + len(num_input_feat_model)
		else:
			self.input_size = input_size

		if output_size_act is None:
			output_size_act = classes_per_cat[concept_name_id]
		self.output_size_act = output_size_act

		# Prefix encoder
		self.encoder = nn.LSTM(
			input_size=self.input_size,
			hidden_size=self.hidden_size,
			num_layers=self.num_layers,
			batch_first=True,
			dropout=dropout,
		)

		# Generator decoder (activity-only output)
		self.activity_embedding = nn.Embedding(self.output_size_act, self.hidden_size)
		self.decoder = nn.LSTM(
			input_size=self.hidden_size * 2,
			hidden_size=self.hidden_size,
			num_layers=self.num_layers,
			batch_first=True,
			dropout=dropout,
		)
		self.generator_head = nn.Linear(self.hidden_size, self.output_size_act)

		# Discriminator (conditional on prefix context)
		self.discriminator_lstm = nn.LSTM(
			input_size=self.hidden_size,
			hidden_size=self.hidden_size,
			num_layers=1,
			batch_first=True,
		)
		self.discriminator_head = nn.Linear(self.hidden_size * 2, 1)

	def _build_prefix_tensor(self, prefixes):
		cats, nums = prefixes

		embedded_cats = [emb(cats[i]) for i, emb in enumerate(self.embeddings)]
		merged_cats = torch.cat(embedded_cats, dim=-1)

		if len(nums):
			merged_nums = torch.cat([num.unsqueeze(2) for num in nums], dim=-1)
		else:
			merged_nums = torch.tensor([], device=merged_cats.device)

		return torch.cat((merged_cats, merged_nums), dim=-1)

	def encode(self, prefixes):
		x = self._build_prefix_tensor(prefixes)
		out, (h, c) = self.encoder(x)
		context = out[:, -1, :]
		return context, (h, c)

	def _get_sos_tokens(self, prefixes):
		cats, _ = prefixes
		return cats[self.concept_name_id][:, -1].long()

	def forward(self, prefixes, target_suffix=None, teacher_forcing_ratio: float = 0.0):
		"""
		Returns activity logits with shape [seq_len_pred, batch, output_size_act].
		"""
		context, (h, c) = self.encode(prefixes)
		batch_size = context.shape[0]
		max_len = self.seq_len_pred

		if target_suffix is not None:
			max_len = target_suffix.shape[1]

		token_t = self._get_sos_tokens(prefixes)
		logits_steps = []

		for t in range(max_len):
			token_emb = self.activity_embedding(token_t).unsqueeze(1)
			context_step = context.unsqueeze(1)
			decoder_in = torch.cat([token_emb, context_step], dim=-1)

			dec_out, (h, c) = self.decoder(decoder_in, (h, c))
			logits_t = self.generator_head(dec_out.squeeze(1))
			logits_steps.append(logits_t.unsqueeze(0))

			if target_suffix is not None and torch.rand(1).item() < teacher_forcing_ratio:
				token_t = target_suffix[:, t].long()
			else:
				token_t = torch.argmax(logits_t, dim=-1)

		return torch.cat(logits_steps, dim=0)

	def discriminate(self, prefixes, suffix_activities):
		"""
		Args:
			suffix_activities:
			  - LongTensor [B, S] activity ids OR
			  - FloatTensor [B, S, C] activity probabilities.

		Returns:
			Tensor [B] with probabilities of being real.
		"""
		context, _ = self.encode(prefixes)

		if suffix_activities.dtype == torch.long:
			suffix_emb = self.activity_embedding(suffix_activities)
		else:
			suffix_emb = suffix_activities @ self.activity_embedding.weight

		_, (h_disc, _) = self.discriminator_lstm(suffix_emb)
		disc_state = h_disc[-1]
		logits = self.discriminator_head(torch.cat([disc_state, context], dim=-1)).squeeze(-1)
		return torch.sigmoid(logits)

	def sample_activity_ids(self, prefixes, max_len: int | None = None):
		logits = self.forward(prefixes=prefixes, target_suffix=None, teacher_forcing_ratio=0.0)
		if max_len is not None and logits.shape[0] != max_len:
			logits = logits[:max_len]
		return torch.argmax(logits, dim=-1).transpose(0, 1).contiguous()

	def beam_search(self, prefixes, beam_width: int = 3, max_len: int | None = None, eos_id: int | None = None):
		"""
		Beam-search decoding for activity suffixes.
		Returns best sequences as LongTensor [B, max_len].
		"""
		context, (h0, c0) = self.encode(prefixes)
		if max_len is None:
			max_len = self.seq_len_pred

		sos_tokens = self._get_sos_tokens(prefixes)
		batch_size = context.shape[0]
		device = context.device
		predictions = []

		for b in range(batch_size):
			context_b = context[b : b + 1]
			h_b = h0[:, b : b + 1, :].contiguous()
			c_b = c0[:, b : b + 1, :].contiguous()
			sos_b = int(sos_tokens[b].item())

			beams = [([], 0.0, sos_b, h_b, c_b, False)]

			for _ in range(max_len):
				candidates = []
				for seq, score, prev_tok, h_prev, c_prev, done in beams:
					if done:
						candidates.append((seq, score, prev_tok, h_prev, c_prev, done))
						continue

					token_tensor = torch.tensor([prev_tok], device=device, dtype=torch.long)
					token_emb = self.activity_embedding(token_tensor).unsqueeze(1)
					decoder_in = torch.cat([token_emb, context_b.unsqueeze(1)], dim=-1)

					dec_out, (h_new, c_new) = self.decoder(decoder_in, (h_prev, c_prev))
					logits = self.generator_head(dec_out.squeeze(1))
					log_probs = F.log_softmax(logits, dim=-1).squeeze(0)

					top_logp, top_idx = torch.topk(log_probs, k=min(beam_width, log_probs.shape[-1]))
					for j in range(top_idx.shape[0]):
						tok = int(top_idx[j].item())
						tok_score = float(top_logp[j].item())
						new_seq = seq + [tok]
						is_done = eos_id is not None and tok == eos_id
						candidates.append((new_seq, score + tok_score, tok, h_new.clone(), c_new.clone(), is_done))

				candidates.sort(key=lambda x: x[1], reverse=True)
				beams = candidates[:beam_width]

			best_seq = beams[0][0]
			if len(best_seq) < max_len:
				best_seq = best_seq + [eos_id if eos_id is not None else 0] * (max_len - len(best_seq))
			predictions.append(torch.tensor(best_seq[:max_len], device=device, dtype=torch.long))

		return torch.stack(predictions, dim=0)

	def save(self, path: str):
		checkpoint = {
			"model_state_dict": self.state_dict(),
			"kwargs": {
				"data_set_categories": self.data_set_categories,
				"model_feat": self.model_feat,
				"concept_name_id": self.concept_name_id,
				"hidden_size": self.hidden_size,
				"num_layers": self.num_layers,
				"seq_len_pred": self.seq_len_pred,
				"input_size": self.input_size,
				"output_size_act": self.output_size_act,
				"dropout": self.dropout,
			},
		}
		return torch.save(checkpoint, path)

	@staticmethod
	def load(path: str):
		checkpoint = torch.load(path, weights_only=False, map_location=torch.device("cpu"))
		model = TaymouriAdversarialLSTM(**checkpoint["kwargs"])
		model.load_state_dict(checkpoint["model_state_dict"])
		return model

