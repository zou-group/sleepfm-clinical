import torch
import torch.nn.functional as F
from torch import nn
from loguru import logger
from collections import OrderedDict
from einops import rearrange
import math
from typing import List, Tuple, Mapping
import torch.nn.utils.rnn as rnn_utils


class Tokenizer(nn.Module):
    def __init__(self, input_size=640, output_size=128):
        super(Tokenizer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.tokenizer = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(4),
            nn.ELU(),
            nn.LayerNorm([4, self.input_size//2]),  
            
            nn.Conv1d(4, 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(8),
            nn.ELU(),
            nn.LayerNorm([8, self.input_size//4]),
            
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.LayerNorm([16, self.input_size//8]),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.LayerNorm([32, self.input_size//16]),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.LayerNorm([64, self.input_size//32]),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.LayerNorm([128, self.input_size//64]),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, self.output_size)
        )

    def forward(self, x):
        
        B, C, T = x.shape
        x = x.view(B, C, -1, self.input_size)
        x = x.permute(0, 1, 2, 3).contiguous().view(-1, 1, self.input_size)
        x = self.tokenizer(x)
        x = x.view(B, C, -1, self.output_size)
        
        return x


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, num_heads=1, dropout=0.1):
        super(AttentionPooling, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x, key_padding_mask=None):
        batch_size, seq_len, input_dim = x.size()
        
        if key_padding_mask is not None:
            if key_padding_mask.size(1) == 1:
                return x.mean(dim=1)
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.to(dtype=torch.bool)
        
            transformer_output = self.transformer_layer(x, src_key_padding_mask=key_padding_mask)
            
            # Invert mask (1 for valid, 0 for padding) and handle the hidden dimension
            attention_mask = (~key_padding_mask).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
            
            # Calculate masked mean
            pooled_output = (transformer_output * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
        else:
            transformer_output = self.transformer_layer(x)
            pooled_output = transformer_output.mean(dim=1)

        return pooled_output


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x


class SetTransformer(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_heads, num_layers, pooling_head=4, dropout=0.1, max_seq_length=128):
        super(SetTransformer, self).__init__()
        # self.patch_embedding = PatchEmbeddingLinear(in_channels, patch_size, embed_dim)
        self.patch_embedding = Tokenizer(input_size=patch_size, output_size=embed_dim)

        self.spatial_pooling = AttentionPooling(embed_dim, num_heads=pooling_head, dropout=dropout)

        self.positional_encoding = PositionalEncoding(max_seq_length, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.temporal_pooling = AttentionPooling(embed_dim, num_heads=pooling_head, dropout=dropout)

    def forward(self, x, mask):

        x = self.patch_embedding(x)
        B, C, S, E = x.shape
        x = rearrange(x, 'b c s e -> (b s) c e')

        mask = mask.unsqueeze(1).expand(-1, S, -1)
        mask = rearrange(mask, 'b t c -> (b t) c')

        if mask.dtype != torch.bool:
            mask = mask.to(dtype=torch.bool)

        x = self.spatial_pooling(x, mask)
        x = x.view((B, S, E))

        x = self.positional_encoding(x)
        x = self.layer_norm(x)

        x = self.transformer_encoder(x)
        embedding = x.clone()
        x = self.temporal_pooling(x)
        return x, embedding



class SleepEventLSTMClassifier(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_classes, pooling_head=4, dropout=0.1, max_seq_length=128):
        super(SleepEventLSTMClassifier, self).__init__()
        
        # Define spatial pooling
        self.spatial_pooling = AttentionPooling(embed_dim, num_heads=pooling_head, dropout=dropout)

        # Set max sequence length
        if max_seq_length is None:
            max_seq_length = 20000
            
        self.positional_encoding = PositionalEncoding(max_seq_length, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Transformer encoder for spatial modeling
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # LSTM for temporal modeling
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim//2, num_layers=num_layers, batch_first=True, dropout=lstm_dropout, bidirectional=True)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x, mask):
        B, C, S, E = x.shape
        
        # Rearrange for spatial pooling
        x = rearrange(x, 'b c s e -> (b s) c e')
        
        # Prepare the mask for spatial pooling
        mask_spatial = mask[:, :, 0]
        mask_spatial = mask_spatial.unsqueeze(1).expand(-1, S, -1)
        mask_spatial = rearrange(mask_spatial, 'b t c -> (b t) c')
        
        # Ensure the mask is boolean
        if mask_spatial.dtype != torch.bool:
            mask_spatial = mask_spatial.to(dtype=torch.bool)

        # Apply spatial pooling
        x = self.spatial_pooling(x, mask_spatial)
        
        # Reshape to (B, S, E) after pooling
        x = x.view(B, S, E)

        # Apply positional encoding and layer normalization
        x = self.positional_encoding(x)
        x = self.layer_norm(x)

        # Apply transformer encoder for spatial modeling
        mask_temporal = mask[:, 0, :]
        x = self.transformer_encoder(x, src_key_padding_mask=mask_temporal)

        # Apply LSTM for temporal modeling
        x, _ = self.lstm(x)  # Shape: (B, S, E)

        # Apply the final fully connected layer for classification
        x = self.fc(x)  # Shape: (B, S, num_classes)

        return x, mask[:, 0, :]  # Return mask along temporal dimension


class DiagnosisFinetuneFullLSTMCOXPHWithDemo(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_classes, pooling_head=4, dropout=0.1, max_seq_length=128):
        super(DiagnosisFinetuneFullLSTMCOXPHWithDemo, self).__init__()
        self.spatial_pooling = AttentionPooling(embed_dim, num_heads=pooling_head, dropout=dropout)

        if max_seq_length is None:
            max_seq_length = 20000

        self.positional_encoding = PositionalEncoding(max_seq_length, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim, 
            hidden_size=embed_dim // 2, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )

        # Linear layer to embed demographic features
        self.demo_embedding = nn.Linear(2, embed_dim // 4)  # Embedding size can be adjusted

        # Update the input size of the disease head to account for embedded demo features
        self.disease_heads = nn.Linear(embed_dim + embed_dim // 4, num_classes)  # embed_dim // 4 from demo features

    def forward(self, x, mask, demo_features):
        B, C, S, E = x.shape
        
        # Rearrange for spatial pooling
        x = rearrange(x, 'b c s e -> (b s) c e')

        mask_spatial = mask[:, :, 0]
        mask_spatial = mask_spatial.unsqueeze(1).expand(-1, S, -1)
        mask_spatial = rearrange(mask_spatial, 'b t c -> (b t) c')

        # Ensure the mask is boolean
        mask_spatial = mask_spatial.to(dtype=torch.bool)

        # Apply spatial pooling
        x = self.spatial_pooling(x, mask_spatial)
        x = x.view(B, S, E)

        # Rearrange mask for temporal pooling
        mask_temporal = mask[:, 0, :]  # Use the mask along the temporal dimension

        # Compute the lengths for packing based on the mask
        lengths = (mask_temporal == 0).sum(dim=1).cpu()

        # Pack the sequence for LSTM
        packed_x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Pass the packed sequence through the LSTM
        packed_out, _ = self.lstm(packed_x)

        # Unpack the sequence
        x, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)

        # Apply mean pooling only over valid lengths
        x = torch.stack([x[i, :lengths[i]].mean(dim=0) for i in range(B)])

        # Embed the demo features
        demo_embed = self.demo_embedding(demo_features)  # (B, embed_dim // 4)

        # Concatenate the demo features with the LSTM output
        x = torch.cat([x, demo_embed], dim=1)  # (B, embed_dim + embed_dim // 4)

        # Apply the disease head to get hazards
        hazards = self.disease_heads(x)
        
        return hazards


class DiagnosisFinetuneDemoOnlyEmbed(nn.Module):
    def __init__(self, embed_dim, num_classes, dropout=0.3):
        super(DiagnosisFinetuneDemoOnlyEmbed, self).__init__()

        self.demo_embedding = nn.Sequential(
            nn.Linear(4, embed_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.disease_heads = nn.Linear(embed_dim // 4, num_classes)

    def forward(self, demo_features):
        demo_embed = self.demo_embedding(demo_features)
        hazards = self.disease_heads(demo_embed)
        return hazards


class DiagnosisFullSupervisedLSTMCOXPHWithDemoEmbed(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_heads, num_layers, num_classes,
                 pooling_head=4, dropout=0.1, max_seq_length=128):
        super(DiagnosisFullSupervisedLSTMCOXPHWithDemoEmbed, self).__init__()

        # Patch embedding layer (Tokenizer)
        self.patch_embedding = Tokenizer(input_size=patch_size, output_size=embed_dim)
        self.spatial_pooling = AttentionPooling(embed_dim, num_heads=pooling_head, dropout=dropout)

        # LSTM for temporal sequence modeling
        self.lstm = nn.LSTM(
            input_size=embed_dim, 
            hidden_size=embed_dim // 2, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )

        # Linear layer for demographic features
        # self.demo_embedding = nn.Linear(2, embed_dim // 4)

        self.demo_embedding = nn.Sequential(
            nn.Linear(2, embed_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Disease head for final prediction
        self.disease_heads = nn.Linear(embed_dim + embed_dim // 4, num_classes)
        self.patch_size = patch_size

    def forward(self, x, mask, demo_features):
        # Step 1: Patch Embedding
        x = self.patch_embedding(x)  # Input: raw data -> Output: (B, C, S, E)
        B, C, S, E = x.shape

        # Step 2: Spatial Pooling
        x = rearrange(x, 'b c s e -> (b s) c e')
        mask_spatial = mask[:, :, 0]
        mask_spatial = mask_spatial.unsqueeze(1).expand(-1, S, -1)
        mask_spatial = rearrange(mask_spatial, 'b t c -> (b t) c').to(dtype=torch.bool)
        x = self.spatial_pooling(x, mask_spatial)
        x = x.view(B, S, E)

        # Step 3: Temporal Processing (LSTM)
        x, _ = self.lstm(x) 
        x = x.mean(dim=1)

        # Step 4: Demographic Embedding
        demo_embed = self.demo_embedding(demo_features)  # Shape: (B, embed_dim // 4)

        # Step 5: Concatenate and Predict
        x = torch.cat([x, demo_embed], dim=1)  # Shape: (B, embed_dim + embed_dim // 4)
        hazards = self.disease_heads(x)
        return hazards


