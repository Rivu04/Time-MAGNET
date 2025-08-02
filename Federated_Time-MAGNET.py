# Import necessary libraries for data processing, machine learning, and visualization
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from peft import get_peft_model, LoraConfig, TaskType
import datetime as dt
import random
from sklearn.metrics import f1_score, confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import math
import warnings
import copy
import time
import gc
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import precision_recall_curve, roc_curve, auc
warnings.filterwarnings('ignore')

# Set random seed for reproducibility across random number generators
def set_seed(seed=42):
    """
    Sets random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): Random seed value, default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# Copy data from source to working directory
src = "/content/mex/data"  # Source directory for dataset
dst = "/content/data"  # Destination directory for dataset
if os.path.exists(dst):
    shutil.rmtree(dst)  # Remove existing destination directory if it exists
shutil.copytree(src, dst)  # Copy directory tree
print("All files and subdirectories copied successfully!")

# Reformat timestamps and add headers to CSV files
base_time = dt.datetime.strptime('2019-02-20 14:29:00', '%Y-%m-%d %H:%M:%S')
directories = [
    '/content/data/act',  # Accelerometer thigh data directory
    '/content/data/acw',  # Accelerometer wrist data directory
    '/content/data/dc_0.05_0.05',  # Depth camera data directory
    '/content/data/pm_1.0_1.0'  # Pressure mat data directory
]

def parse_mm_ss_t(timestamp_str):
    """
    Parse timestamp strings in mm:ss.t format to datetime objects.

    Args:
        timestamp_str (str): Timestamp string in mm:ss.t format.

    Returns:
        datetime or None: Parsed datetime object or None if parsing fails.
    """
    try:
        parts = timestamp_str.split(':')
        minutes = int(parts[0])
        seconds = float(parts[1])
        delta = dt.timedelta(minutes=minutes, seconds=seconds)
        return base_time + delta
    except:
        return None

def reformat_timestamp(timestamp_str):
    """
    Reformat timestamps to a consistent format (%Y-%m-%d %H:%M:%S.%f).

    Args:
        timestamp_str (str): Input timestamp string in various possible formats.

    Returns:
        str or None: Reformatted timestamp string or None if parsing fails.
    """
    formats = [
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%d-%m-%Y %H:%M:%S',
        '%d-%m-%Y %H:%M',
        '%d/%m/%Y %H:%M:%S',
        '%d/%m/%Y %H:%M:%S.%f'
    ]
    for fmt in formats:
        try:
            parsed = dt.datetime.strptime(timestamp_str, fmt)
            return parsed.strftime('%Y-%m-%d %H:%M:%S.%f')
        except:
            continue
    parsed = parse_mm_ss_t(timestamp_str)
    if parsed:
        return parsed.strftime('%Y-%m-%d %H:%M:%S.%f')
    return None


# Process each directory and its subfolders to reformat CSV files
for directory in directories:
    print(f"\nProcessing directory: {directory}")
    subfolders = [f for f in os.listdir(directory)
                  if os.path.isdir(os.path.join(directory, f))
                  and f in [f"{i:02d}" for i in range(1, 31)]]
    if not subfolders:
        print(f" No subfolders (01–30) found.")
        continue
    for subfolder in sorted(subfolders):
        subfolder_path = os.path.join(directory, subfolder)
        print(f" Processing subfolder: {subfolder}")
        csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
        if not csv_files:
            print(f" No CSV files found in {subfolder_path}")
            continue
        for file in csv_files:
            file_path = os.path.join(subfolder_path, file)
            try:
                # Read CSV without header, assume first column is timestamp
                df = pd.read_csv(file_path, header=None, dtype={0: str})
                new_timestamps = []
                for ts in df[0]:
                    reformatted = reformat_timestamp(ts.strip())
                    if reformatted is None:
                        raise ValueError(f"Unparseable timestamp: {ts}")
                    new_timestamps.append(reformatted)
                df[0] = new_timestamps
                # Assign column names based on modality
                if 'act' in directory or 'acw' in directory:
                    df.columns = ['timestamp', 'x', 'y', 'z']  # Shape: [N, 4]
                else:
                    num_features = df.shape[1] - 1
                    df.columns = ['timestamp'] + [f'feature_{i+1}' for i in range(num_features)]
                    # Shape: [N, 193] for dc, [N, 513] for pm
                df.to_csv(file_path, header=True, index=False)
            except Exception as e:
                print(f" Error processing {file}: {e}")
print("\n✅ All CSV files updated with reformatted timestamps and proper headers.")


# Configuration parameters for data processing and model
window = 5  # 5-second windows for data segmentation
increment = 1  # 1-second increment for overlapping windows
ac_frames_per_second = 100  # Sampling rate for accelerometer data
dc_frames_per_second = 15  # Sampling rate for depth camera data
pm_frames_per_second = 15  # Sampling rate for pressure mat data
ac_max_length = ac_frames_per_second * window  # 500 frames for accelerometer
dc_max_length = dc_frames_per_second * window  # 75 frames for depth camera
pm_max_length = pm_frames_per_second * window  # 75 frames for pressure mat
num_classes = 7  # Number of activity classes
emb_dim = 512  # Embedding dimension for model
accumulation_steps = 6  # Gradient accumulation steps for training


# T5 configuration for time-series encoder
Time_T5_config = T5Config(
    vocab_size=4096,
    d_model=emb_dim,  # Model dimension: 512
    d_kv=64,  # Dimension of key/value in attention
    d_ff=2048,  # Feed-forward dimension
    num_layers=8,  # Number of transformer layers
    num_heads=8,  # Number of attention heads
    relative_attention_num_buckets=32,
    dropout_rate=0.1,
    layer_norm_epsilon=1e-6,
    initializer_factor=1.0,
    feed_forward_proj="gated-gelu",
    is_encoder_decoder=False,
    use_cache=False,
    pad_token_id=0,
    eos_token_id=1,
)


# Enhanced RMSNorm with learnable scaling
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        """
        Implements RMS Normalization with learnable scaling.

        Args:
            dim (int): Dimension of the input tensor.
            eps (float): Small value for numerical stability, default is 1e-6.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Shape: [dim]
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor, shape [batch_size, seq_length, dim].

        Returns:
            torch.Tensor: Normalized tensor, same shape as input.
        """
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# SwiGLU activation function
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        """
        Implements SwiGLU activation function.

        Args:
            dim (int): Input dimension.
            hidden_dim (int, optional): Hidden dimension, defaults to dim * 8/3.
        """
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8/3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Shape: [dim, hidden_dim]
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)  # Shape: [dim, hidden_dim]
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)  # Shape: [hidden_dim, dim]
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor, shape [batch_size, seq_length, dim].

        Returns:
            torch.Tensor: Output tensor, same shape as input.
        """
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        hidden = self.dropout(hidden)
        return self.w3(hidden)


# Expert Network with skip connections
class ExpertNetwork(nn.Module):
    def __init__(self, dim, hidden_dim):
        """
        Implements an expert network with skip connections.

        Args:
            dim (int): Input and output dimension.
            hidden_dim (int): Hidden dimension for the feed-forward network.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Shape: [dim, hidden_dim]
        self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=False)  # Shape: [hidden_dim, hidden_dim]
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)  # Shape: [hidden_dim, dim]
      
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
      
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor, shape [batch_size * seq_length, dim].

        Returns:
            torch.Tensor: Output tensor, same shape as input.
        """
        h1 = F.gelu(self.w1(x))
        h1 = self.norm1(h1)
        h2 = F.gelu(self.w2(h1))
        h2 = self.norm2(h2)
        h2 = self.dropout(h2)
      
        return self.w3(h2 + h1)  # Skip connection


# Mixture of Experts with load balancing
class MixtureOfExperts(nn.Module):
    def __init__(self, dim, num_experts=8, top_k=2, hidden_dim=None):
        """
        Implements Mixture of Experts with top-k routing and load balancing.

        Args:
            dim (int): Input and output dimension.
            num_experts (int): Number of expert networks, default is 8.
            top_k (int): Number of experts to select, default is 2.
            hidden_dim (int, optional): Hidden dimension, defaults to dim * 4.
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        hidden_dim = hidden_dim or dim * 4
      
        self.gate = nn.Linear(dim, num_experts, bias=False)  # Shape: [dim, num_experts]
      
        self.experts = nn.ModuleList([
            ExpertNetwork(dim, hidden_dim) for _ in range(num_experts)
        ])
      
        self.noise_std = 0.1
        self.load_balancing_loss_weight = 0.01
        # Store the load balancing loss for retrieval
        self.load_balancing_loss = 0.0
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor, shape [batch_size, seq_length, dim].

        Returns:
            torch.Tensor: Output tensor, same shape as input.
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # Shape: [batch_size * seq_length, dim]

        # Gating mechanism with noise for training
        gate_logits = self.gate(x_flat)  # Shape: [batch_size * seq_length, num_experts]
        if self.training:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise

        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)  # Shape: [batch_size * seq_length, top_k]

        # Load balancing loss computation
        gates_softmax = F.softmax(gate_logits, dim=-1)
        expert_usage = torch.mean(gates_softmax, dim=0)

        # Entropy-based load balancing loss
        entropy_loss = -torch.sum(expert_usage * torch.log(expert_usage + 1e-8))
        uniform_entropy = -torch.log(torch.tensor(1.0 / self.num_experts))

        # Store load balancing loss (normalized)
        self.load_balancing_loss = self.load_balancing_loss_weight * (uniform_entropy - entropy_loss)

        # Expert computation - compute all experts first
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x_flat))  # Shape: [batch_size * seq_length, dim]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # Shape: [batch_size * seq_length, num_experts, dim]

        # Combine expert outputs using proper indexing
        final_output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            gate_weight = top_k_gates[:, i].unsqueeze(-1)

            # Use gather to select the appropriate expert outputs
            selected_experts = expert_outputs.gather(1, expert_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, D)).squeeze(1)
            final_output += gate_weight * selected_experts
          
        return final_output.view(B, T, D)  # Shape: [batch_size, seq_length, dim]


# Graph Attention Layer with weighted adjacency
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        """
        Implements a single graph attention layer.

        Args:
            in_features (int): Input feature dimension.
            out_features (int): Output feature dimension.
            dropout (float): Dropout rate, default is 0.1.
            alpha (float): LeakyReLU negative slope, default is 0.2.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
      
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))  # Shape: [in_features, out_features]
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))  # Shape: [2 * out_features, 1]
        self.bias = nn.Parameter(torch.zeros(out_features))  # Shape: [out_features]
      
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
      
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.layer_norm = nn.LayerNorm(out_features)
    
    def forward(self, h, adj):
        """
        Args:
            h (torch.Tensor): Input node features, shape [batch_size, num_nodes, in_features].
            adj (torch.Tensor): Adjacency matrix, shape [batch_size, num_nodes, num_nodes].

        Returns:
            torch.Tensor: Output node features, shape [batch_size, num_nodes, out_features].
        """
        Wh = torch.matmul(h, self.W)  # Shape: [batch_size, num_nodes, out_features]
        B, N, _ = Wh.shape

        # Self-attention mechanism
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # Shape: [batch_size, num_nodes, 1]
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  # Shape: [batch_size, num_nodes, 1]
        e = self.leakyrelu(Wh1 + Wh2.transpose(-1, -2))  # Shape: [batch_size, num_nodes, num_nodes]

        # Mask for sparsity (if adj == 0)
        mask = adj > 0
        e_masked = torch.where(mask, e, torch.tensor(-1e9, dtype=e.dtype, device=e.device))

        # Softmax
        attention = F.softmax(e_masked, dim=-1)

        # Incorporate edge weights
        attention = attention * adj

        # Renormalize
        attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-6)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh) + self.bias  # Shape: [batch_size, num_nodes, out_features]
        h_prime = self.layer_norm(h_prime)
      
        return h_prime


# Multi-Head Graph Attention
class MultiHeadGraphAttention(nn.Module):
    def __init__(self, in_features, out_features, num_heads=8, dropout=0.1):
        """
        Implements multi-head graph attention mechanism.

        Args:
            in_features (int): Input feature dimension.
            out_features (int): Output feature dimension.
            num_heads (int): Number of attention heads, default is 8.
            dropout (float): Dropout rate, default is 0.1.
        """
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.head_dim = out_features // num_heads
      
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, self.head_dim, dropout)
            for _ in range(num_heads)
        ])
      
        self.out_proj = nn.Linear(out_features, out_features)  # Shape: [out_features, out_features]
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)
    
    def forward(self, x, adj):
        """
        Args:
            x (torch.Tensor): Input node features, shape [batch_size, num_nodes, in_features].
            adj (torch.Tensor): Adjacency matrix, shape [batch_size, num_nodes, num_nodes].

        Returns:
            torch.Tensor: Output node features, shape [batch_size, num_nodes, out_features].
        """
        head_outputs = []
        for attention in self.attentions:
            head_outputs.append(attention(x, adj))  # Shape: [batch_size, num_nodes, head_dim]
          
        concat_output = torch.cat(head_outputs, dim=-1)  # Shape: [batch_size, num_nodes, out_features]
        output = self.out_proj(concat_output)
        output = self.dropout(output)
        output = self.layer_norm(output + x)  # Residual connection
      
        return output


# Fusion Layer-II for multimodal processing
class FusionLayer(nn.Module):
    def __init__(self, hidden_dim, num_modalities):
        """
        Implements a fusion layer combining graph attention, MoE, and SwiGLU.

        Args:
            hidden_dim (int): Hidden dimension for processing.
            num_modalities (int): Number of input modalities (e.g., 4 for act, acw, dc, pm).
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Graph attention
        self.graph_attention = MultiHeadGraphAttention(
            hidden_dim, hidden_dim, num_heads=8, dropout=0.1
        )

        # Mixture of experts
        self.moe = MixtureOfExperts(hidden_dim, num_experts=4, top_k=2)

        # SwiGLU feedforward
        self.swiglu = SwiGLU(hidden_dim)

        # RMSNorm layers
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
        self.norm3 = RMSNorm(hidden_dim)

        # Learnable residual scaling
        self.residual_scale = nn.Parameter(torch.ones(3))  # Shape: [3]
    
    def forward(self, x, adj):
        """
        Args:
            x (torch.Tensor): Input tensor, shape [batch_size, num_modalities, hidden_dim].
            adj (torch.Tensor): Adjacency matrix, shape [batch_size, num_modalities, num_modalities].

        Returns:
            torch.Tensor: Fused output, same shape as input.
        """
        # Graph attention with residual connection
        residual = x
        x = self.norm1(x)
        x = self.graph_attention(x, adj)
        x = residual + self.residual_scale[0] * x

        # Mixture of experts with residual connection
        residual = x
        x = self.norm2(x)
        x = self.moe(x)
        x = residual + self.residual_scale[1] * x

        # SwiGLU feedforward with residual connection
        residual = x
        x = self.norm3(x)
        x = self.swiglu(x)
        x = residual + self.residual_scale[2] * x
      
        return x


# This module contains MAGNET (Multimodal Adaptive Graph Neural Expert Transformer)
class EnhancedFusionModule(nn.Module):
    def __init__(self, input_dim=emb_dim, hidden_dim=emb_dim, num_modalities=4, num_layers=3,
                 learnable_adj=True, dynamic_adj=True, directed=False, hierarchy=None):
        """
        Implements Multimodal Adaptive Graph Neural Expert Transformer (MAGNET).

        Args:
            input_dim (int): Input dimension, default is 512.
            hidden_dim (int): Hidden dimension, default is 512.
            num_modalities (int): Number of modalities, default is 4.
            num_layers (int): Number of fusion layers, default is 3.
            learnable_adj (bool): Whether to learn adjacency matrix, default is True.
            dynamic_adj (bool): Whether to compute dynamic adjacency, default is True.
            directed (bool): Whether graph is directed, default is False (not used).
            hierarchy (torch.Tensor, optional): Hierarchy mask, shape [num_modalities, num_modalities] (not used).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.num_layers = num_layers

        # Extension flags
        self.learnable_adj = learnable_adj
        self.dynamic_adj = dynamic_adj
        self.directed = directed
        self.hierarchy = hierarchy

        # Input projections
        self.input_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),  # Shape: [input_dim, hidden_dim]
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(num_modalities)
        ])

        # Graph attention
        self.graph_attention = MultiHeadGraphAttention(
            in_features=hidden_dim,
            out_features=hidden_dim,
            num_heads=8,
            dropout=0.1
        )

        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            FusionLayer(hidden_dim, num_modalities) for _ in range(num_layers)
        ])

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # Shape: [hidden_dim, hidden_dim]
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)  # Shape: [hidden_dim, input_dim]
        )
                   
        self.final_norm = RMSNorm(input_dim)

        # For learnable adjacency (sigmoid-gated matrix)
        if self.learnable_adj:
            self.adj_logits = nn.Parameter(torch.zeros(num_modalities, num_modalities))  # Shape: [num_modalities, num_modalities]
    
    def create_adjacency_matrix(self, projected_embeddings):
        """
        Creates adjacency matrix for graph attention.

        Args:
            projected_embeddings (list): List of modality embeddings, each shape [batch_size, hidden_dim].

        Returns:
            torch.Tensor: Adjacency matrix, shape [batch_size, num_modalities, num_modalities].
        """
        batch_size = projected_embeddings[0].shape[0]
        device = projected_embeddings[0].device
        M = self.num_modalities

        # Base adjacency
        if self.dynamic_adj:
            # Dynamic: Compute from cosine similarity
            emb_stack = torch.stack(projected_embeddings, dim=1)  # Shape: [batch_size, num_modalities, hidden_dim]
            emb_norm = F.normalize(emb_stack, p=2, dim=-1)
            cos_sim = emb_norm @ emb_norm.transpose(1, 2)  # Shape: [batch_size, num_modalities, num_modalities]
            adj = (cos_sim + 1.0) / 2.0
          
            if not self.directed:
                adj = (adj + adj.transpose(1, 2)) / 2.0
              
        else:
            # Static base
            adj = torch.ones(batch_size, M, M, device=device)

        # Learnable: Multiply by sigmoid-gated weights
        if self.learnable_adj:
            learn_adj = torch.sigmoid(self.adj_logits).unsqueeze(0).expand(batch_size, -1, -1)
            adj = adj * learn_adj

        # Add self-loops
        eye = torch.eye(M, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        adj = adj + 0.5 * eye

        # For hierarchical relationships/subgraphs: Apply a mask or weights
        if self.hierarchy is not None:
            if self.hierarchy.dim() == 2:
                hierarchy_mask = self.hierarchy.unsqueeze(0).expand(batch_size, -1, -1).to(device)
            else:
                hierarchy_mask = self.hierarchy.to(device)
            adj = adj * hierarchy_mask
          
        return adj
    
    def forward(self, modality_embeddings):
        """
        Args:
            modality_embeddings (list): List of modality embeddings, each shape [batch_size, input_dim].

        Returns:
            torch.Tensor: Fused output, shape [batch_size, input_dim].
        """
        batch_size = modality_embeddings[0].shape[0]
        device = modality_embeddings[0].device

        # Project inputs
        projected_embeddings = []
        for i, embedding in enumerate(modality_embeddings):
            projected = self.input_projections[i](embedding)  # Shape: [batch_size, hidden_dim]
            projected_embeddings.append(projected)

        # Create adjacency (now dynamic/learnable)
        adj = self.create_adjacency_matrix(projected_embeddings)

        # Stack modalities
        x = torch.stack(projected_embeddings, dim=1)  # Shape: [batch_size, num_modalities, hidden_dim]

        # Graph attention (Fusion Layer-I)
        attended_x = self.graph_attention(x, adj)
        x = x + attended_x

        # Multi-layer fusion (Fusion Layer-II)
        for layer in self.fusion_layers:
            x = layer(x, adj)

        # Global pooling (Fusion Layer-III)
        attention_weights = F.softmax(x.mean(dim=-1), dim=-1)  # Shape: [batch_size, num_modalities]
        fused_output = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)  # Shape: [batch_size, hidden_dim]

        # Final projection (Fusion Layer-III)
        output = self.output_projection(fused_output)  # Shape: [batch_size, input_dim]
        output = self.final_norm(output)
      
        return output


# Customized T5 Time Series Encoder
class T5_TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim=3, config=Time_T5_config):
        """
        Implements a T5-based encoder for time-series data.

        Args:
            input_dim (int): Input dimension (e.g., 3 for x, y, z accelerometer data).
            config (T5Config): T5 model configuration.
        """
        super().__init__()
        self.config = config
      
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, config.d_model // 2),  # Shape: [input_dim, d_model//2]
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.d_model // 2, config.d_model)  # Shape: [d_model//2, d_model]
        )

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(ac_max_length, config.d_model))  # Shape: [500, d_model]

        # Enhanced encoder with LoRA
        self.encoder = T5Stack(config, embed_tokens=None)
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "k", "v"]
        )
      
        self.encoder = get_peft_model(self.encoder, lora_config)

        # Multi-scale pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),  # Shape: [d_model*2, d_model]
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor, shape [batch_size, seq_length, input_dim].

        Returns:
            torch.Tensor: Encoded output, shape [batch_size, d_model].
        """
        B, T, C = x.shape
        x = self.input_projection(x)  # Shape: [batch_size, seq_length, d_model]

        # Add positional encoding
        x = x + self.positional_encoding[:T].unsqueeze(0)

        # Attention mask
        attention_mask = torch.ones(B, T, device=x.device, dtype=torch.long)

        # Encode
        encoder_outputs = self.encoder(
            inputs_embeds=x,
            attention_mask=attention_mask
        )
      
        hidden_states = encoder_outputs.last_hidden_state  # Shape: [batch_size, seq_length, d_model]

        # Multi-scale pooling
        avg_pooled = self.global_pool(hidden_states.transpose(1, 2)).squeeze(-1)  # Shape: [batch_size, d_model]
        max_pooled = self.max_pool(hidden_states.transpose(1, 2)).squeeze(-1)  # Shape: [batch_size, d_model]

        # Combine pooled features
        combined = torch.cat([avg_pooled, max_pooled], dim=-1)  # Shape: [batch_size, d_model*2]
        output = self.output_projection(combined)  # Shape: [batch_size, d_model]
      
        return output


# DART-CNN Encoder for depth camera and pressure mat
class DART_CNNEncoder(nn.Module):
    def __init__(self, input_shape, emb_dim=768):
        """
        Implements DART-CNN encoder with dual attention for 2D data.

        Args:
            input_shape (tuple): Input shape (height, width), e.g., (12, 16) for dc, (32, 16) for pm.
            emb_dim (int): Output embedding dimension, default is 768.
        """
        super(DART_CNNEncoder, self).__init__()
        height, width = input_shape

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # Shape: [1, 64, height, width]
        self.bn1 = nn.BatchNorm2d(64)
      
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Shape: [64, 128, height, width]
        self.bn2 = nn.BatchNorm2d(128)
      
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # Shape: [128, 256, height/2, width/2]
        self.bn3 = nn.BatchNorm2d(256)
      
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)  # Shape: [256, 512, height/2, width/2]
        self.bn4 = nn.BatchNorm2d(512)

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),  # Shape: [512, 1, height/2, width/2]
            nn.Sigmoid()
        )

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512 // 16),  # Shape: [512, 32]
            nn.ReLU(),
            nn.Linear(512 // 16, 512),  # Shape: [32, 512]
            nn.Sigmoid()
        )
      
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Enhanced fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512, emb_dim),  # Shape: [512, emb_dim]
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(emb_dim, emb_dim)  # Shape: [emb_dim, emb_dim]
        )

        # Temporal processing with LSTM, RNN, and GRU
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=emb_dim // 2,
            num_layers=3,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
      
        self.rnn = nn.RNN(
            input_size=emb_dim,
            hidden_size=emb_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
      
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=emb_dim // 2,
            num_layers=1,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor, shape [batch_size, seq_length, height, width].

        Returns:
            torch.Tensor: Encoded output, shape [batch_size, emb_dim].
        """
        B, T, H, W = x.shape
        x = x.view(B * T, 1, H, W)  # Shape: [batch_size * seq_length, 1, height, width]

        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
      
        x = F.max_pool2d(x, 2)  # Shape: [batch_size * seq_length, 128, height/2, width/2]
      
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))  # Shape: [batch_size * seq_length, 512, height/2, width/2]

        # Attention mechanisms
        spatial_att = self.spatial_attention(x)
        channel_att = self.channel_attention(x).view(B * T, 512, 1, 1)
        x = x * spatial_att * channel_att

        # Global pooling and FC
        x = self.adaptive_pool(x)  # Shape: [batch_size * seq_length, 512, 1, 1]
        x = self.flatten(x)  # Shape: [batch_size * seq_length, 512]
        x = self.fc(x)  # Shape: [batch_size * seq_length, emb_dim]

        # Reshape for temporal processing
        x = x.view(B, T, -1)  # Shape: [batch_size, seq_length, emb_dim]

        # Temporal processing with LSTM, RNN, and GRU
        lstm_out, _ = self.lstm(x)  # Shape: [batch_size, seq_length, emb_dim]
        rnn_out, _ = self.rnn(lstm_out)  # Shape: [batch_size, seq_length, emb_dim]
        gru_out, _ = self.gru(rnn_out)  # Shape: [batch_size, seq_length, emb_dim]

        # Combine with residual connection
        x = x + gru_out

        # Global temporal pooling
        return x.mean(dim=1)  # Shape: [batch_size, emb_dim]

# Main Activity Classification Model
class ActivityClassificationModel(nn.Module):
    def __init__(self, hidden_dim=emb_dim, num_classes=7):
        """
        Implements the main activity classification model with multimodal encoders and fusion.

        Args:
            hidden_dim (int): Hidden dimension, default is 512.
            num_classes (int): Number of output classes, default is 7.
        """
        super().__init__()
        # Modality specific encoders
        self.act_encoder = T5_TimeSeriesEncoder(input_dim=3)  # For accelerometer thigh
        self.acw_encoder = T5_TimeSeriesEncoder(input_dim=3)  # For accelerometer wrist
        self.dc_encoder = DART_CNNEncoder(input_shape=(12, 16), emb_dim=hidden_dim)  # For depth camera
        self.pm_encoder = DART_CNNEncoder(input_shape=(32, 16), emb_dim=hidden_dim)  # For pressure mat

        # MAGNET fusion module
        self.fusion = EnhancedFusionModule(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_modalities=4,
            num_layers=3,
            learnable_adj=True,
            dynamic_adj=True,
            directed=False,    # Not used in the code and paper
            hierarchy=None     # Not used in the code and paper
        )

        # Multi Layer Hierarchical classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # Shape: [hidden_dim, hidden_dim]
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.4),
          
            nn.Linear(hidden_dim, hidden_dim // 2),  # Shape: [hidden_dim, hidden_dim//2]
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
          
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # Shape: [hidden_dim//2, hidden_dim//4]
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(0.2),
          
            nn.Linear(hidden_dim // 4, num_classes)  # Shape: [hidden_dim//4, num_classes]
        )

        # Learnable missing embedding
        self.missing_embedding = nn.Parameter(torch.randn(1, hidden_dim))  # Shape: [1, hidden_dim]

        # Modality-specific learnable weights
        self.modality_weights = nn.Parameter(torch.ones(4))  # Shape: [4]
    
    def forward(self, act_tensor, acw_tensor, dc_tensor, pm_tensor):
        """
        Args:
            act_tensor (torch.Tensor): Accelerometer thigh data, shape [batch_size, 500, 3] or None.
            acw_tensor (torch.Tensor): Accelerometer wrist data, shape [batch_size, 500, 3] or None.
            dc_tensor (torch.Tensor): Depth camera data, shape [batch_size, 75, 12, 16] or None.
            pm_tensor (torch.Tensor): Pressure mat data, shape [batch_size, 75, 32, 16] or None.

        Returns:
            tuple:
                - logits (torch.Tensor): Classification logits, shape [batch_size, num_classes].
                - act_emb (torch.Tensor): Accelerometer thigh embeddings, shape [batch_size, hidden_dim].
                - acw_emb (torch.Tensor): Accelerometer wrist embeddings, shape [batch_size, hidden_dim].
                - dc_emb (torch.Tensor): Depth camera embeddings, shape [batch_size, hidden_dim].
                - pm_emb (torch.Tensor): Pressure mat embeddings, shape [batch_size, hidden_dim].
        """
        embeddings = []
        batch_size = (act_tensor.shape[0] if act_tensor is not None else
                      acw_tensor.shape[0] if acw_tensor is not None else
                      dc_tensor.shape[0] if dc_tensor is not None else
                      pm_tensor.shape[0])

        # Encode modalities
        if act_tensor is not None:
            act_emb = self.act_encoder(act_tensor)  # Shape: [batch_size, hidden_dim]
        else:
            act_emb = self.missing_embedding.expand(batch_size, -1)
        embeddings.append(act_emb * self.modality_weights[0])
      
        if acw_tensor is not None:
            acw_emb = self.acw_encoder(acw_tensor)  # Shape: [batch_size, hidden_dim]
        else:
            acw_emb = self.missing_embedding.expand(batch_size, -1)
        embeddings.append(acw_emb * self.modality_weights[1])
      
        if dc_tensor is not None:
            dc_emb = self.dc_encoder(dc_tensor)  # Shape: [batch_size, hidden_dim]
        else:
            dc_emb = self.missing_embedding.expand(batch_size, -1)
        embeddings.append(dc_emb * self.modality_weights[2])
      
        if pm_tensor is not None:
            pm_emb = self.pm_encoder(pm_tensor)  # Shape: [batch_size, hidden_dim]
        else:
            pm_emb = self.missing_embedding.expand(batch_size, -1)
        embeddings.append(pm_emb * self.modality_weights[3])

        # MAGNET fusion
        fused_features = self.fusion(embeddings)  # Shape: [batch_size, hidden_dim]

        # Classifier
        logits = self.classifier(fused_features)  # Shape: [batch_size, num_classes]
      
        return logits, act_emb, acw_emb, dc_emb, pm_emb
    
    def get_moe_load_balancing_loss(self):
        """
        Computes the total MoE load balancing loss across all MoE modules.

        Returns:
            float: Average MoE load balancing loss.
        """
        total_loss = 0.0
        count = 0
      
        for module in self.modules():
            if isinstance(module, MixtureOfExperts):
                total_loss += module.load_balancing_loss
                count += 1
              
        return total_loss / count if count > 0 else 0.0

# Custom Dataset for Sensor Data
class SensorDataset(Dataset):
    def __init__(self, act_dir, acw_dir, dc_dir, pm_dir, participant_ids):
        """
        Initializes dataset for multimodal sensor data.

        Args:
            act_dir (str): Directory for accelerometer thigh data.
            acw_dir (str): Directory for accelerometer wrist data.
            dc_dir (str): Directory for depth camera data.
            pm_dir (str): Directory for pressure mat data.
            participant_ids (list): List of participant IDs (e.g., ['01', '02']).
        """
        self.data = []
        skipped_samples = 0
        for person in sorted(participant_ids):
            act_path = os.path.join(act_dir, person)
            acw_path = os.path.join(acw_dir, person)
            dc_path = os.path.join(dc_dir, person)
            pm_path = os.path.join(pm_dir, person)
            if not all(os.path.exists(p) for p in [act_path, acw_path, dc_path, pm_path]):
                print(f"Missing directories for participant {person}")
                continue
            act_files = sorted([f for f in os.listdir(act_path) if f.endswith('.csv')])
            for af in act_files:
                try:
                    activity_id = int(af.split('_')[0])
                    label = activity_id - 1  # Zero-based indexing
                    aw = af.replace("_act_", "_acw_")
                    dc = af.replace("_act_", "_dc_")
                    pm = af.replace("_act_", "_pm_")
                    act_file = os.path.join(act_path, af)
                    acw_file = os.path.join(acw_path, aw)
                    dc_file = os.path.join(dc_path, dc)
                    pm_file = os.path.join(pm_path, pm)
                    act_data = load_sensor_file(act_file) if os.path.exists(act_file) else None
                    acw_data = load_sensor_file(acw_file) if os.path.exists(acw_file) else None
                    dc_data = load_sensor_file(dc_file) if os.path.exists(dc_file) else None
                    pm_data = load_sensor_file(pm_file) if os.path.exists(pm_file) else None
                    if all(d is None for d in [act_data, acw_data, dc_data, pm_data]):
                        print(f"No valid data for {af}")
                        skipped_samples += 1
                        continue
                    time_windows = split_windows(act_data, acw_data, dc_data, pm_data)
                    if len(time_windows) == 0:
                        print(f"No valid time windows for {af}")
                        skipped_samples += 1
                        continue
                    for window_data in time_windows:
                        act_window, acw_window, dc_window, pm_window = window_data
                        act_tensor = torch.tensor(act_window, dtype=torch.float32) if act_window is not None else None
                        acw_tensor = torch.tensor(acw_window, dtype=torch.float32) if acw_window is not None else None
                        dc_tensor = torch.tensor(dc_window, dtype=torch.float32) if dc_window is not None else None
                        pm_tensor = torch.tensor(pm_window, dtype=torch.float32) if pm_window is not None else None
                        self.data.append((act_tensor, acw_tensor, dc_tensor, pm_tensor, label))
                except Exception as e:
                    print(f"Error processing {af}: {e}")
                    skipped_samples += 1
                    continue
        print(f"Skipped {skipped_samples} samples due to errors.")
    
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (act_tensor, acw_tensor, dc_tensor, pm_tensor, label)
                - act_tensor: Shape [500, 3] or None
                - acw_tensor: Shape [500, 3] or None
                - dc_tensor: Shape [75, 12, 16] or None
                - pm_tensor: Shape [75, 32, 16] or None
                - label: Integer class label
        """
        return self.data[idx]

  
def load_sensor_file(filepath):
    """
    Loads and validates a sensor data CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        np.ndarray or None: Sensor data array or None if loading fails.
            - act/acw: Shape [N, 4] (timestamp, x, y, z)
            - dc: Shape [N, 193] (timestamp, 192 features)
            - pm: Shape [N, 513] (timestamp, 512 features)
    """
    try:
        df = pd.read_csv(filepath)
        if 'pm' in filepath and df.shape[1] != 513:
            raise ValueError(f"Expected 513 columns in {filepath}, got {df.shape[1]}")
          
        if 'dc' in filepath and df.shape[1] != 193:
            raise ValueError(f"Expected 193 columns in {filepath}, got {df.shape[1]}")
          
        if ('act' in filepath or 'acw' in filepath) and df.shape[1] != 4:
            raise ValueError(f"Expected 4 columns in {filepath}, got {df.shape[1]}")
          
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
      
        if df['timestamp'].isna().any():
            raise ValueError(f"Invalid timestamps in {filepath}")
          
        df = df.dropna(subset=['timestamp']).reset_index(drop=True)
        df['timestamp'] = df['timestamp'].astype(np.int64) // 10**9

        # Validate numerical columns
        feature_columns = [col for col in df.columns if col != 'timestamp']
      
        if not all(df[feature_columns].apply(lambda x: pd.to_numeric(x, errors='coerce').notna()).all()):
            raise ValueError(f"Non-numeric values found in feature columns of {filepath}")
          
        return df.values
      
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
      
    except ValueError as e:
        print(f"ValueError in {filepath}: {e}")
        return None
      
    except Exception as e:
        print(f"Unexpected error loading {filepath}: {e}")
        return None


def augment_data(data):
    """
    Applies Gaussian noise to data for augmentation.

    Args:
        data (np.ndarray): Input data array.

    Returns:
        np.ndarray: Augmented data with same shape as input.
    """
    noise = np.random.normal(0, 0.01, data.shape)
    return data + noise


def normalize_data(data):
    """
    Normalizes data using mean and standard deviation.

    Args:
        data (np.ndarray): Input data array.

    Returns:
        np.ndarray: Normalized data with same shape as input.
    """
    data_mean = np.mean(data, axis=0, keepdims=True)
    data_std = np.std(data, axis=0, keepdims=True)
    return (data - data_mean) / (data_std + 1e-8)


def resample_data(data, target_length):
    """
    Resamples data to a target length using linear interpolation.

    Args:
        data (np.ndarray): Input data, shape [N, ...].
        target_length (int): Desired length of the first dimension.

    Returns:
        np.ndarray: Resampled data, shape [target_length, ...].
    """
    if data is None or len(data) == 0:
        if len(data.shape) == 2:
            return np.zeros((target_length, data.shape[1]))
        else:
            return np.zeros((target_length,) + data.shape[1:])
          
    if len(data) == target_length:
        return data
      
    if len(data) == 1:
        return np.repeat(data, target_length, axis=0)
      
    x = np.linspace(0, 1, len(data))
    x_new = np.linspace(0, 1, target_length)
  
    if len(data.shape) == 2:
        interpolated = np.zeros((target_length, data.shape[1]))
        for i in range(data.shape[1]):
            interpolator = interp1d(x, data[:, i], kind='linear', fill_value='extrapolate')
            interpolated[:, i] = interpolator(x_new)
        return interpolated
    else:
        interpolated = np.zeros((target_length,) + data.shape[1:])
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                interpolator = interp1d(x, data[:, i, j], kind='linear', fill_value='extrapolate')
                interpolated[:, i, j] = interpolator(x_new)
              
        return interpolated


def split_windows(act_data, acw_data, dc_data, pm_data):
    """
    Splits sensor data into time windows.

    Args:
        act_data (np.ndarray): Accelerometer thigh data, shape [N, 4] or None.
        acw_data (np.ndarray): Accelerometer wrist data, shape [N, 4] or None.
        dc_data (np.ndarray): Depth camera data, shape [N, 193] or None.
        pm_data (np.ndarray): Pressure mat data, shape [N, 513] or None.

    Returns:
        list: List of windows, each containing [act_window, acw_window, dc_window, pm_window].
            - act_window: Shape [500, 3] or None
            - acw_window: Shape [500, 3] or None
            - dc_window: Shape [75, 12, 16] or None
            - pm_window: Shape [75, 32, 16] or None
    """
    if all(data is None for data in [act_data, acw_data, dc_data, pm_data]):
        return []
      
    start_times = []
    end_times = []
  
    for data in [act_data, acw_data, dc_data, pm_data]:
        if data is not None and len(data) > 0:
            start_times.append(data[0, 0])
            end_times.append(data[-1, 0])
          
    if not start_times or not end_times:
        return []
      
    start_time = max(start_times)
    end_time = min(end_times)
  
    if start_time >= end_time:
        return []
      
    windows = []
    current_time = start_time
  
    while current_time + window <= end_time:
        window_end = current_time + window
      
        act_window = None
        if act_data is not None and len(act_data) > 0:
            act_mask = (act_data[:, 0] >= current_time) & (act_data[:, 0] < window_end)
            if act_mask.sum() > 0:
                act_window = act_data[act_mask, 1:4]  # Shape: [N, 3]
              
        acw_window = None
        if acw_data is not None and len(acw_data) > 0:
            acw_mask = (acw_data[:, 0] >= current_time) & (acw_data[:, 0] < window_end)
            if acw_mask.sum() > 0:
                acw_window = acw_data[acw_mask, 1:4]  # Shape: [N, 3]
              
        dc_window = None
        if dc_data is not None and len(dc_data) > 0:
            dc_mask = (dc_data[:, 0] >= current_time) & (dc_data[:, 0] < window_end)
            if dc_mask.sum() > 0:
                dc_window_flat = dc_data[dc_mask, 1:]  # Shape: [N, 192]
                if dc_window_flat.shape[1] == 192:
                    dc_window = dc_window_flat.reshape(-1, 12, 16)  # Shape: [N, 12, 16]
                  
        pm_window = None
        if pm_data is not None and len(pm_data) > 0:
            pm_mask = (pm_data[:, 0] >= current_time) & (pm_data[:, 0] < window_end)
            if pm_mask.sum() > 0:
                pm_window_flat = pm_data[pm_mask, 1:]  # Shape: [N, 512]
                if pm_window_flat.shape[1] == 512:
                    pm_window = pm_window_flat.reshape(-1, 32, 16)  # Shape: [N, 32, 16]
                  
        if all(w is None for w in [act_window, acw_window, dc_window, pm_window]):
            current_time += increment
            continue
          
        if act_window is not None and len(act_window) > 0:
            act_window = resample_data(act_window, ac_max_length)  # Shape: [500, 3]
            act_window = augment_data(normalize_data(act_window))
          
        if acw_window is not None and len(acw_window) > 0:
            acw_window = resample_data(acw_window, ac_max_length)  # Shape: [500, 3]
            acw_window = augment_data(normalize_data(acw_window))
          
        if dc_window is not None and len(dc_window) > 0:
            dc_window = resample_data(dc_window, dc_max_length)  # Shape: [75, 12, 16]
            dc_window = normalize_data(dc_window.reshape(-1, 12*16)).reshape(-1, 12, 16)
          
        if pm_window is not None and len(pm_window) > 0:
            pm_window = resample_data(pm_window, pm_max_length)  # Shape: [75, 32, 16]
            pm_window = normalize_data(pm_window.reshape(-1, 32*16)).reshape(-1, 32, 16)
          
        windows.append([act_window, acw_window, dc_window, pm_window])
        current_time += increment
      
    return windows


def collate_fn(batch):
    """
    Collates a batch of samples into tensors, handling missing modalities.

    Args:
        batch (list): List of tuples (act_tensor, acw_tensor, dc_tensor, pm_tensor, label).

    Returns:
        tuple:
            - act_batch (torch.Tensor): Shape [batch_size, 500, 3] or None.
            - acw_batch (torch.Tensor): Shape [batch_size, 500, 3] or None.
            - dc_batch (torch.Tensor): Shape [batch_size, 75, 12, 16] or None.
            - pm_batch (torch.Tensor): Shape [batch_size, 75, 32, 16] or None.
            - label_batch (torch.Tensor): Shape [batch_size], dtype torch.long.
    """
    act_list, acw_list, dc_list, pm_list, labels = [], [], [], [], []
    for act, acw, dc, pm, y in batch:
        act_list.append(act)
        acw_list.append(acw)
        dc_list.append(dc)
        pm_list.append(pm)
        labels.append(y)
      
    act_batch = None
    acw_batch = None
    dc_batch = None
    pm_batch = None
  
    valid_acts = [x for x in act_list if x is not None]
  
    if valid_acts:
        act_batch = torch.stack(valid_acts)  # Shape: [batch_size, 500, 3]
    valid_acws = [x for x in acw_list if x is not None]
  
    if valid_acws:
        acw_batch = torch.stack(valid_acws)  # Shape: [batch_size, 500, 3]
    valid_dcs = [x for x in dc_list if x is not None]
  
    if valid_dcs:
        dc_batch = torch.stack(valid_dcs)  # Shape: [batch_size, 75, 12, 16]
    valid_pms = [x for x in pm_list if x is not None]
  
    if valid_pms:
        pm_batch = torch.stack(valid_pms)  # Shape: [batch_size, 75, 32, 16]
    label_batch = torch.tensor(labels, dtype=torch.long)  # Shape: [batch_size]
  
    if all(batch is None for batch in [act_batch, acw_batch, dc_batch, pm_batch]):
        raise ValueError("All modalities are None for the batch")
      
    return act_batch, acw_batch, dc_batch, pm_batch, label_batch


def training_step_with_moe_loss(model, act_x, acw_x, dc_x, pm_x, y, criterion):
    """
    Performs a single training step with MoE load balancing loss.

    Args:
        model (nn.Module): Activity classification model.
        act_x (torch.Tensor): Accelerometer thigh data, shape [batch_size, 500, 3] or None.
        acw_x (torch.Tensor): Accelerometer wrist data, shape [batch_size, 500, 3] or None.
        dc_x (torch.Tensor): Depth camera data, shape [batch_size, 75, 12, 16] or None.
        pm_x (torch.Tensor): Pressure mat data, shape [batch_size, 75, 32, 16] or None.
        y (torch.Tensor): Labels, shape [batch_size].
        criterion (nn.Module): Loss function (CrossEntropyLoss).

    Returns:
        tuple:
            - total_loss (torch.Tensor): Total loss (classification + MoE).
            - logits (torch.Tensor): Model predictions, shape [batch_size, num_classes].
            - classification_loss (torch.Tensor): Classification loss.
            - load_balancing_loss (float): MoE load balancing loss.
    """
    logits, _, _, _, _ = model(act_x, acw_x, dc_x, pm_x)  # Shape: [batch_size, num_classes]
    classification_loss = criterion(logits, y)
    load_balancing_loss = model.get_moe_load_balancing_loss()
    total_loss = classification_loss + load_balancing_loss
  
    return total_loss, logits, classification_loss, load_balancing_loss


def evaluate_with_moe_loss(model, loader, device, store_embeddings=False):
    """
    Evaluates the model on a dataset, tracking MoE loss and embeddings.

    Args:
        model (nn.Module): Activity classification model.
        loader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to run the model on.
        store_embeddings (bool): Whether to store modality embeddings, default is False.

    Returns:
        tuple:
            - accuracy (float): Classification accuracy.
            - weighted_f1 (float): Weighted F1 score.
            - avg_loss (float): Average loss.
            - prec (float): Weighted precision.
            - rec (float): Weighted recall.
            - act_embeddings (np.ndarray): Shape [N, hidden_dim] or None.
            - acw_embeddings (np.ndarray): Shape [N, hidden_dim] or None.
            - dc_embeddings (np.ndarray): Shape [N, hidden_dim] or None.
            - pm_embeddings (np.ndarray): Shape [N, hidden_dim] or None.
            - all_labels (np.ndarray): Shape [N].
            - all_probs (np.ndarray): Shape [N, num_classes].
            - cm (np.ndarray): Confusion matrix, shape [num_classes, num_classes].
    """
    model.eval()
  
    total, correct = 0, 0
    all_preds, all_labels = [], []
    all_probs = []
    act_embeddings, acw_embeddings, dc_embeddings, pm_embeddings = [], [], [], []
    total_moe_loss = 0.0
    num_batches = 0
    total_loss = 0.0
  
    with torch.no_grad():
        for batch_idx, (act_x, acw_x, dc_x, pm_x, y) in enumerate(tqdm(loader, desc="Evaluating Test Set")):
            act_x = act_x.to(device) if act_x is not None else None
            acw_x = acw_x.to(device) if acw_x is not None else None
            dc_x = dc_x.to(device) if dc_x is not None else None
            pm_x = pm_x.to(device) if pm_x is not None else None
          
            y = y.to(device)
          
            logits, act_emb, acw_emb, dc_emb, pm_emb = model(act_x, acw_x, dc_x, pm_x)
            probs = F.softmax(logits, dim=1)  # Shape: [batch_size, num_classes]
            preds = torch.argmax(logits, dim=1)  # Shape: [batch_size]
          
            moe_loss = model.get_moe_load_balancing_loss()
            total_moe_loss += moe_loss if isinstance(moe_loss, (int, float)) else moe_loss.item()
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item()
            num_batches += 1
            correct += (preds == y).sum().item()
            total += y.size(0)
          
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
          
            if store_embeddings:
                act_embeddings.extend(act_emb.cpu().numpy())
                acw_embeddings.extend(acw_emb.cpu().numpy())
                dc_embeddings.extend(dc_emb.cpu().numpy())
                pm_embeddings.extend(pm_emb.cpu().numpy())
              
    accuracy = correct / total
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    prec, rec, _, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    cm = confusion_matrix(all_labels, all_preds)  # Shape: [num_classes, num_classes]
    avg_moe_loss = total_moe_loss / num_batches
    avg_loss = total_loss / num_batches
  
    class_report = classification_report(all_labels, all_preds, digits=3)
    with open('/content/classification_report.txt', 'w') as f:
        f.write(class_report)
    print("\nClassification Report saved to '/content/classification_report.txt'")
  
    print("Classification Report:\n", class_report)
    print("Confusion Matrix:\n", cm)
    print("Per-Class F1 Scores:", per_class_f1)
    print(f"Average MoE Load Balancing Loss: {avg_moe_loss:.4f}")
  
    if store_embeddings:
        return accuracy, weighted_f1, avg_loss, prec, rec, np.array(act_embeddings), np.array(acw_embeddings), np.array(dc_embeddings), np.array(pm_embeddings), np.array(all_labels), np.array(all_probs), cm
    else:
        return accuracy, weighted_f1, avg_loss, prec, rec, None, None, None, None, np.array(all_labels), np.array(all_probs), cm


# Federated Learning Class
class FederatedActivityClassifier:
    def __init__(self, client_datasets, unified_val_dataset, device):
        """
        Initializes federated learning classifier.

        Args:
            client_datasets (dict): Dictionary of client ID to SensorDataset.
            unified_val_dataset (SensorDataset): Unified validation dataset.
            device (torch.device): Device to run the model on.
        """
        self.global_model = ActivityClassificationModel(hidden_dim=emb_dim, num_classes=7)
      
        self.client_datasets = client_datasets
        self.unified_val_dataset = unified_val_dataset
        self.device = device
        class_counts = np.zeros(num_classes)
        all_labels = []
      
        for ds in self.client_datasets.values():
            labels = [y for _, _, _, _, y in ds]
            all_labels.extend(labels)
          
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        class_counts[unique_labels] = counts
        class_weights = 1.0 / (class_counts + 1e-6)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)  # Shape: [num_classes]
    
    def save_model(self, path):
        """
        Saves the global model state dictionary.

        Args:
            path (str): Path to save the model.
        """
        torch.save(self.global_model.state_dict(), path)
    
    def load_model(self, path):
        """
        Loads the global model state dictionary.

        Args:
            path (str): Path to the model state dictionary.
        """
        self.global_model.load_state_dict(torch.load(path))
    
    def local_update(self, local_model, dataset, learning_rate=1e-4, epochs=1, batch_size=8):
        """
        Performs local training on a client's dataset.

        Args:
            local_model (nn.Module): Local model instance.
            dataset (SensorDataset): Client's dataset.
            learning_rate (float): Learning rate, default is 1e-4.
            epochs (int): Number of local epochs, default is 1.
            batch_size (int): Batch size, default is 8.

        Returns:
            dict: Updated local model state dictionary.
        """
        if dataset is None or len(dataset) == 0:
            return copy.deepcopy(local_model.state_dict())
          
        local_model.to(self.device)
      
        local_model.train()
      
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        optimizer = AdamW(local_model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
        scaler = GradScaler()
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
      
        for epoch in range(epochs):
            running_loss = 0.0
            num_batches = 0
            optimizer.zero_grad()
            for batch_idx, (act_x, acw_x, dc_x, pm_x, y) in enumerate(tqdm(dataloader, desc=f"Local Training Epoch {epoch+1}")):
                act_x = act_x.to(self.device) if act_x is not None else None
                acw_x = acw_x.to(self.device) if acw_x is not None else None
                dc_x = dc_x.to(self.device) if dc_x is not None else None
                pm_x = pm_x.to(self.device) if pm_x is not None else None
                y = y.to(self.device)
              
                with autocast(device_type='cuda'):
                    total_loss, _, _, _ = training_step_with_moe_loss(
                        local_model, act_x, acw_x, dc_x, pm_x, y, criterion
                    )
                  
                    total_loss = total_loss / accumulation_steps
                  
                scaler.scale(total_loss).backward()
              
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                  
                running_loss += total_loss.item() * accumulation_steps
                num_batches += 1
              
            scheduler.step()
          
        local_model.to('cpu')
        state_dict = copy.deepcopy(local_model.state_dict())
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
      
        return state_dict
    
    def evaluate_model(self, model, loader, dataset_name="", store_embeddings=False):
        """
        Evaluates the model on a dataset.

        Args:
            model (nn.Module): Model to evaluate.
            loader (DataLoader): DataLoader for the dataset.
            dataset_name (str): Name of the dataset for logging.
            store_embeddings (bool): Whether to store embeddings, default is False.

        Returns:
            dict: Metrics including loss, accuracy, precision, recall, f1, and embeddings.
        """
        if loader is None:
            print(f"No data available for {dataset_name}")
            return {
                'loss': 0,
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'act_embeddings': None,
                'acw_embeddings': None,
                'dc_embeddings': None,
                'pm_embeddings': None
            }
          
        acc, f1, avg_loss, prec, rec, act_emb, acw_emb, dc_emb, pm_emb, labels, probs, cm = evaluate_with_moe_loss(
            model, loader, self.device, store_embeddings
        )
      
        metrics = {
            'loss': avg_loss,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'act_embeddings': act_emb,
            'acw_embeddings': acw_emb,
            'dc_embeddings': dc_emb,
            'pm_embeddings': pm_emb,
            'labels': labels,
            'probs': probs,
            'cm': cm
        }
      
        return metrics
    
    def average_state_dicts(self, state_dicts):
        """
        Averages multiple state dictionaries for federated learning.

        Args:
            state_dicts (list): List of state dictionaries from clients.

        Returns:
            dict: Averaged state dictionary.
        """
        if not state_dicts:
            return None
          
        avg_state = copy.deepcopy(state_dicts[0])
      
        for key in avg_state:
            if avg_state[key].dtype.is_floating_point:
                for i in range(1, len(state_dicts)):
                    avg_state[key] += state_dicts[i][key]
                avg_state[key] /= len(state_dicts)
              
        return avg_state
    
    def federated_training(self, global_epochs=10, local_epochs=5, learning_rate=1e-4, batch_size=8, num_clients_per_round=9, patience=6, save_path='/content/best_global_model.pth'):
        """
        Performs federated training across clients.

        Args:
            global_epochs (int): Number of global epochs, default is 10.
            local_epochs (int): Number of local epochs per client, default is 5.
            learning_rate (float): Learning rate, default is 1e-4.
            batch_size (int): Batch size, default is 8.
            num_clients_per_round (int): Number of clients per round, default is 9.
            patience (int): Patience for early stopping, default is 6.
            save_path (str): Path to save the best model.

        Returns:
            tuple: Lists of train_losses, val_metrics_history, train_accuracies, val_accuracies.
                - train_losses (list): Average training loss per epoch.
                - val_metrics_history (list): List of validation metrics dictionaries.
                - train_accuracies (list): Training accuracies per epoch.
                - val_accuracies (list): Validation accuracies per epoch.
        """
        print("Starting Federated Training")
        available_clients = list(self.client_datasets.keys())
        if len(available_clients) < 2:
            print("Not enough clients with training data")
            return [], [], []
        print(f"Available clients: {available_clients}")
      
        train_losses = []
        val_metrics_history = []
        train_accuracies = []
        val_accuracies = []
        best_val_acc = float('inf')
        patience_counter = 0
        best_model_found = False
      
        for epoch in range(global_epochs):
            start_time = time.time()
            selected_clients = random.sample(available_clients, min(num_clients_per_round, len(available_clients)))
            print(f"\nEpoch {epoch + 1}, Selected clients: {selected_clients}")
          
            client_updates = []
            total_train_loss = 0
            num_train_batches = 0
            train_correct = 0
            train_total = 0
          
            for client in selected_clients:
                print(f"Training client: {client}")
                local_model = ActivityClassificationModel(hidden_dim=emb_dim, num_classes=7)
                local_model.load_state_dict(copy.deepcopy(self.global_model.state_dict()))
              
                updated_state = self.local_update(
                    local_model,
                    self.client_datasets[client],
                    learning_rate=learning_rate,
                    epochs=local_epochs,
                    batch_size=batch_size
                )
              
                if updated_state is not None:
                    client_updates.append(updated_state)
                  
                local_model.to(self.device)
              
                local_model.eval()
              
                train_dataloader = DataLoader(self.client_datasets[client], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
              
                criterion = nn.CrossEntropyLoss(weight=self.class_weights)
              
                with torch.no_grad():
                    for act_x, acw_x, dc_x, pm_x, y in tqdm(train_dataloader, desc=f"Local Training Loss Computation {client}"):
                        act_x = act_x.to(self.device) if act_x is not None else None  # Shape: [batch_size, 500, 3] or None
                        acw_x = acw_x.to(self.device) if acw_x is not None else None  # Shape: [batch_size, 500, 3] or None
                        dc_x = dc_x.to(self.device) if dc_x is not None else None  # Shape: [batch_size, 75, 12, 16] or None
                        pm_x = pm_x.to(self.device) if pm_x is not None else None  # Shape: [batch_size, 75, 32, 16] or None
                      
                        y = y.to(self.device)  # Shape: [batch_size]
                      
                        total_loss, logits, _, _ = training_step_with_moe_loss(
                            local_model, act_x, acw_x, dc_x, pm_x, y, criterion
                        )  # logits: [batch_size, num_classes]
                      
                        preds = torch.argmax(logits, dim=1)  # Shape: [batch_size]
                        train_correct += (preds == y).sum().item()
                        train_total += y.size(0)
                        total_train_loss += total_loss.item()
                        num_train_batches += 1
                      
                local_model.train()
              
                del local_model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
              
            if client_updates:
                avg_state = self.average_state_dicts(client_updates)
                self.global_model.load_state_dict(avg_state)
                del avg_state
                del client_updates
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
            else:
                print("No client updates received in this round")
                continue
            avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0
            train_acc = train_correct / train_total if train_total > 0 else 0
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_acc)
            self.global_model.to(self.device)
          
            val_metrics = {}
            if self.unified_val_dataset is not None:
                val_loader = DataLoader(self.unified_val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
                val_metrics = self.evaluate_model(self.global_model, val_loader, "Unified Validation")
            else:
                print("No unified validation dataset available.")
                val_metrics = {'loss': float('inf'), 'accuracy': 0, 'f1': 0}
              
            val_metrics_history.append(val_metrics)
            val_accuracies.append(val_metrics.get('accuracy', 0))
            epoch_time = time.time() - start_time
          
            # Log epoch results including training and validation metrics
            print(f"\nEpoch {epoch+1}/{global_epochs} - Time: {epoch_time:.2f}s")
            print(f" Train Loss: {avg_train_loss:.4f}")
            print(f" Train Acc: {train_acc:.4f}")
            print(f" Val Loss: {val_metrics.get('loss', 0):.4f}")
            print(f" Val Acc: {val_metrics.get('accuracy', 0):.4f}, F1: {val_metrics.get('f1', 0):.4f}")
          
            # Check for early stopping based on validation loss
            if val_metrics.get('accuracy', float('inf')) < best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                self.save_model(save_path)
                best_model_found = True
                print(f" Best model saved!")
            else:
                patience_counter += 1
                print(f" No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            print("-" * 80)
            # Move model to CPU and clear GPU memory
            self.global_model.to('cpu')
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            print(f"CPU memory cleared after epoch {epoch+1}")
          
        # Load best model for final evaluation if it exists
        if best_model_found and os.path.exists(save_path):
            print(f"Loading best model from {save_path} for final evaluation.")
            self.load_model(save_path)
        elif not best_model_found:
            print("No improvement during training. Skipping loading of best model.")
        else:
            print(f"Saved model not found at {save_path}. Skipping loading.")
        self.global_model.to('cpu')
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        return train_losses, val_metrics_history, train_accuracies, val_accuracies


# Visualization function for training and evaluation results
def create_visualizations(train_losses, val_losses, train_accuracies, val_accuracies, act_embeddings, acw_embeddings, dc_embeddings, pm_embeddings, all_labels, all_probs, cm):
    """
    Creates visualizations including loss curves, accuracy curves, t-SNE, UMAP, ROC-AUC, precision-recall, and confusion matrix.

    Args:
        train_losses (list): Training losses per epoch.
        val_losses (list): Validation losses per epoch.
        train_accuracies (list): Training accuracies per epoch.
        val_accuracies (list): Validation accuracies per epoch.
        act_embeddings (np.ndarray): Accelerometer thigh embeddings, shape [N, hidden_dim].
        acw_embeddings (np.ndarray): Accelerometer wrist embeddings, shape [N, hidden_dim].
        dc_embeddings (np.ndarray): Depth camera embeddings, shape [N, hidden_dim].
        pm_embeddings (np.ndarray): Pressure mat embeddings, shape [N, hidden_dim].
        all_labels (np.ndarray): Ground truth labels, shape [N].
        all_probs (np.ndarray): Predicted probabilities, shape [N, num_classes].
        cm (np.ndarray): Confusion matrix, shape [num_classes, num_classes].
    """
    sns.set_palette("tab10")
  
    # Plot training and validation loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(False)
    plt.savefig('/content/loss_curve.png')
    plt.close()
  
    # Plot training and validation accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(val_accuracies, label='Validation Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(False)
    plt.savefig('/content/metrics_curve.png')
    plt.close()
  
    # Compute fused embeddings by averaging modality embeddings
    fused_embeddings = (act_embeddings + acw_embeddings + dc_embeddings + pm_embeddings) / 4  # Shape: [N, hidden_dim]
    # t-SNE visualization of fused embeddings
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_embeddings = tsne.fit_transform(fused_embeddings)  # Shape: [N, 2]
  
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=all_labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Activity Class')
    plt.title('t-SNE Visualization of Test Set Fused Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(False)
    plt.savefig('/content/tsne_plot.png')
    plt.close()
  
    # UMAP visualization of fused embeddings
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    umap_embeddings = umap_reducer.fit_transform(fused_embeddings)  # Shape: [N, 2]
  
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=all_labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Activity Class')
    plt.title('UMAP Visualization of Test Set Fused Embeddings')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.grid(False)
    plt.savefig('/content/umap_plot.png')
    plt.close()
  
    # ROC-AUC curves for each class
    plt.figure(figsize=(10, 6))
    all_labels_onehot = F.one_hot(torch.tensor(all_labels), num_classes=7).numpy()  # Shape: [N, num_classes]
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(all_labels_onehot[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curves (Test Set)')
    plt.legend()
    plt.grid(False)
    plt.savefig('/content/roc_auc_curve.png')
    plt.close()
  
    # Precision-Recall curves for each class
    plt.figure(figsize=(10, 6))
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(all_labels_onehot[:, i], all_probs[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'Class {i} (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves (Test Set)')
    plt.legend()
    plt.grid(False)
    plt.savefig('/content/precision_recall_curve.png')
    plt.close()
  
    # Confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Test Set)')
    plt.savefig('/content/confusion_matrix.png')
    plt.close()
    print("\n✅ Visualizations saved successfully!")

# Input directories for sensor data
act_dir = '/content/data/act'  # Accelerometer thigh data directory
acw_dir = '/content/data/acw'  # Accelerometer wrist data directory
dc_dir = '/content/data/dc_0.05_0.05'  # Depth camera data directory
pm_dir = '/content/data/pm_1.0_1.0'  # Pressure mat data directory

# Create datasets and dataloaders
all_participants = [f'{i:02d}' for i in range(1, 31)]  # List of participant IDs: ['01', '02', ..., '30']
random.shuffle(all_participants)
  
train_participants = all_participants[:21]  # 70% for training
val_participants = all_participants[21:24]  # 10% for validation
test_participants = all_participants[24:]  # 20% for testing

# Create client datasets for training
client_datasets = {p: SensorDataset(act_dir, acw_dir, dc_dir, pm_dir, [p]) for p in train_participants}

# Create validation and test datasets
val_dataset = SensorDataset(act_dir, acw_dir, dc_dir, pm_dir, val_participants)
test_dataset = SensorDataset(act_dir, acw_dir, dc_dir, pm_dir, test_participants)

# Log dataset sizes
print(f"Dataset sizes - Val: {len(val_dataset)}, Test: {len(test_dataset)}")
for p, ds in client_datasets.items():
    print(f"Client {p}: {len(ds)} samples")
  
# Create DataLoaders
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# Instantiate the model and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Run the federated training
if __name__ == "__main__":
    # Initialize federated learning trainer
    fl_trainer = FederatedActivityClassifier(client_datasets, val_dataset, device)
  
    # Perform federated training
    train_losses, val_metrics_history, train_accuracies, val_accuracies = fl_trainer.federated_training(
        global_epochs=10,
        local_epochs=5,
        num_clients_per_round=9
    )
  
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
  
    # Move global model to device for evaluation
    fl_trainer.global_model.to(device)
  
    # Evaluate on test set with embeddings
    test_metrics = fl_trainer.evaluate_model(fl_trainer.global_model, test_loader, "Test", store_embeddings=True)
  
    # Perform detailed evaluation
    test_acc, test_f1, test_loss, test_precision, test_recall, act_embeddings, acw_embeddings, dc_embeddings, pm_embeddings, test_labels, test_probs, test_cm = evaluate_with_moe_loss(
        fl_trainer.global_model, test_loader, device, store_embeddings=True
    )
  
    # Log final results
    print(f"\n✅ FINAL RESULTS:")
    print(f"Test Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"Test F1 Score: {test_metrics['f1']:.3f}")
    print(f"Test Precision: {test_metrics['precision']:.3f}")
    print(f"Test Recall: {test_metrics['recall']:.3f}")
  
    print(f"Accelerometer Thigh Embeddings Shape: {test_metrics['act_embeddings'].shape}")
    print(f"Accelerometer Wrist Embeddings Shape: {test_metrics['acw_embeddings'].shape}")
    print(f"Depth Camera Embeddings Shape: {test_metrics['dc_embeddings'].shape}")
    print(f"Pressure Mat Embeddings Shape: {test_metrics['pm_embeddings'].shape}")
  
    # Save embeddings as numpy arrays
    np.save('/content/act_embeddings.npy', test_metrics['act_embeddings'])
    np.save('/content/acw_embeddings.npy', test_metrics['acw_embeddings'])
    np.save('/content/dc_embeddings.npy', test_metrics['dc_embeddings'])
    np.save('/content/pm_embeddings.npy', test_metrics['pm_embeddings'])
  
    # Create visualizations
    create_visualizations(
        train_losses,
        [vm['loss'] for vm in val_metrics_history],
        train_accuracies,
        [vm['accuracy'] for vm in val_metrics_history],
        test_metrics['act_embeddings'],
        test_metrics['acw_embeddings'],
        test_metrics['dc_embeddings'],
        test_metrics['pm_embeddings'],
        test_labels,
        test_probs,
        test_cm
    )
  
    print("\n✅ All tasks completed successfully!")
