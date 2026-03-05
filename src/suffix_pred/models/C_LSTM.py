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
        cat_categories, _ = data_set_categories
        
        cat_input_feat_model, num_input_feat_model = model_feat
        
        # Use the first value in the tuple as the key and the second value as the value
        cat_dict = {cat[0]: cat[1] for cat in cat_categories}
        
        # Get input feature sizes to determine the model input size
        list_of_classes_per_cat = [cat_dict[cat_feat] for cat_feat in cat_input_feat_model if cat_feat in cat_dict]
        
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

        # Special LSTM heads for each ouput variable
        self.lstm_act = nn.LSTM(input_size=self.hidden_size,
                                hidden_size=self.hidden_size,
                                batch_first=True,
                                dropout=0.2,
                                num_layers=self.num_layers)

        # Linear output heads:
        self.act_head  = nn.Linear(self.hidden_size, self.output_size_act)
    
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

    def forward(self, input):
        # Build your input: x of shape (T = seq. len., B = batch size, input_size)
        x = self.__input_construction(data=input)      # (T, B, input_size)
        # print(x.shape)

        x = x.permute(1, 0, 2) # (B, T, input_size)

        # Shared LSTM (batch_first=True)
        out_seq, _ = self.shared_lstm(x) # (B, T, hidden_size)
        # print(out_seq.shape)

        # Batch‑norm over features at each time step
        y = out_seq.transpose(1, 2) # (B, hidden_size, T)
        # print(y.shape)
        y = self.bn1(y) # (B, hidden_size, T)
        # print(y.shape)
        y = y.transpose(1, 2) # (B, T, hidden_size)

        # Head LSTMs (batch_first=True); grab only last hidden states
        _, (h_act,  _) = self.lstm_act(y) # h_act: (1, B, hidden_size)
        # print(h_act.shape)
        h_act  = h_act.squeeze(0) # (B, hidden_size)

        # Final heads & activations
        a_logits = self.act_head(h_act) # (B, output_size_act)

        # Transform logits into probabilities
        a_probs = F.softmax(a_logits, dim=-1)
        
        return a_probs    
    
    def save(self, path : str):
        """
        Store the trained model at path.
        """
        checkpoint = {'model_state_dict' : self.state_dict(),
                      'kwargs' : {
                        'data_set_categories' : self.data_set_categories,
                        'hidden_size': self.hidden_size,
                        'num_layers': self.num_layers,
                        'model_feat': self.model_feat,
                        'input_size': self.input_size,
                        'output_size_act': self.output_size_act
                      }
                    }
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
