# Importing necessary libraries
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
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
def set_seed(seed=42):
    """
    Set random seeds for reproducibility across random, numpy, and torch.
    
    Args:
        seed (int): Seed value for random number generators (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# Copy data to working directory
src = "/content/mex/data"
dst = "/content/data"
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print("All files and subdirectories copied successfully!")

# Reformat timestamps and add headers
base_time = dt.datetime.strptime('2019-02-20 14:29:00', '%Y-%m-%d %H:%M:%S')
directories = [
    '/content/data/act',
    '/content/data/acw',
    '/content/data/dc_0.05_0.05',
    '/content/data/pm_1.0_1.0'
]

def parse_mm_ss_t(timestamp_str):
    """
    Parse timestamp in MM:SS.t format to datetime.
    
    Args:
        timestamp_str (str): Timestamp string in MM:SS.t format
    
    Returns:
        datetime: Parsed datetime object or None if parsing fails
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
    Reformat various timestamp formats to standardized format.
    
    Args:
        timestamp_str (str): Input timestamp string
    
    Returns:
        str: Reformatted timestamp in '%Y-%m-%d %H:%M:%S.%f' format or None if parsing fails
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
                df = pd.read_csv(file_path, header=None, dtype={0: str})
                new_timestamps = []
                for ts in df[0]:
                    reformatted = reformat_timestamp(ts.strip())
                    if reformatted is None:
                        raise ValueError(f"Unparseable timestamp: {ts}")
                    new_timestamps.append(reformatted)
                df[0] = new_timestamps
                if 'act' in directory or 'acw' in directory:
                    df.columns = ['timestamp', 'x', 'y', 'z']
                else:
                    num_features = df.shape[1] - 1
                    df.columns = ['timestamp'] + [f'feature_{i+1}' for i in range(num_features)]
                df.to_csv(file_path, header=True, index=False)
            except Exception as e:
                print(f" Error processing {file}: {e}")
print("\n✅ All CSV files updated with reformatted timestamps and proper headers.")


# Enhanced Configuration
window = 5 # 5-second windows
increment = 1 # Increased overlap for more training data
ac_frames_per_second = 100
dc_frames_per_second = 15
pm_frames_per_second = 15
ac_max_length = ac_frames_per_second * window # 500 frames
dc_max_length = dc_frames_per_second * window # 75 frames
pm_max_length = pm_frames_per_second * window # 75 frames
num_classes = 7
emb_dim = 512
accumulation_steps = 6


# Enhanced Time-T5 configuration
Time_T5_config = T5Config(
    vocab_size=4096,
    d_model=emb_dim,
    d_kv=64,
    d_ff=2048,
    num_layers=8,
    num_heads=8,
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
        Root Mean Square Layer Normalization.
        
        Args:
            dim (int): Dimension of the input tensor (embedding dimension)
            eps (float): Small constant to avoid division by zero (default: 1e-6)
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
      
    def forward(self, x):
        """
        Forward pass for RMSNorm.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim)
        
        Returns:
            torch.Tensor: Normalized tensor of same shape
        """
        # Simple RMSNorm without running statistics
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# SwiGLU
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        """
        SwiGLU activation function with dropout.
        
        Args:
            dim (int): Input dimension (embedding dimension)
            hidden_dim (int, optional): Hidden dimension, defaults to int(dim * 8/3)
        """
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8/3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(0.1)
      
    def forward(self, x):
        """
        Forward pass for SwiGLU.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim)
        
        Returns:
            torch.Tensor: Output tensor of same shape
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
        Expert Network with GELU activations, RMSNorm, dropout, and skip connections.
        
        Args:
            dim (int): Input/output dimension (embedding dimension)
            hidden_dim (int): Hidden layer dimension
        """
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
      
        self.norm1 = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
      
    def forward(self, x):
        """
        Forward pass for ExpertNetwork.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size * sequence_length, dim)
        
        Returns:
            torch.Tensor: Output tensor of same shape
        """
        h1 = F.gelu(self.w1(x))
        h1 = self.norm1(h1)
        h2 = F.gelu(self.w2(h1))
        h2 = self.norm2(h2)
        h2 = self.dropout(h2)
        return self.w3(h2 + h1) # Skip connection


# Mixture of Experts with load balancing
class MixtureOfExperts(nn.Module):
    def __init__(self, dim, num_experts=8, top_k=2, hidden_dim=None):
        """
        Mixture of Experts layer with gating, top-k selection, noise during training, and load balancing loss.
        
        Args:
            dim (int): Input dimension (embedding dimension)
            num_experts (int): Number of experts (default: 8)
            top_k (int): Number of top experts to select (default: 2)
            hidden_dim (int, optional): Hidden dimension for experts, defaults to dim * 4
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        if hidden_dim is None:
            hidden_dim = dim * 4
          
        self.gate = nn.Linear(dim, num_experts, bias=False)
      
        self.experts = nn.ModuleList([
            ExpertNetwork(dim, hidden_dim) for _ in range(num_experts)
        ])
      
        self.noise_std = 0.1
        self.load_balancing_loss_weight = 0.01
      
        # Store the load balancing loss for retrieval
        self.load_balancing_loss = 0.0
      
    def forward(self, x):
        """
        Forward pass for MixtureOfExperts.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim)
        
        Returns:
            torch.Tensor: Output tensor of same shape
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)
      
        # Gating mechanism with noise for training
        gate_logits = self.gate(x_flat)
        if self.training:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise
          
        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
      
        # Load balancing loss computation
        gates_softmax = F.softmax(gate_logits, dim=-1)
        expert_usage = torch.mean(gates_softmax, dim=0) # Average usage per expert
      
        # Entropy-based load balancing loss
        entropy_loss = -torch.sum(expert_usage * torch.log(expert_usage + 1e-8))
        uniform_entropy = -torch.log(torch.tensor(1.0 / self.num_experts))
      
        # Store load balancing loss (normalized)
        self.load_balancing_loss = self.load_balancing_loss_weight * (uniform_entropy - entropy_loss)
      
        # Expert computation - compute all experts first
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x_flat))
        expert_outputs = torch.stack(expert_outputs, dim=1) # Shape: [B*T, num_experts, D]
      
        # Combine expert outputs using proper indexing
        final_output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i] # Shape: [B*T]
            gate_weight = top_k_gates[:, i].unsqueeze(-1) # Shape: [B*T, 1]
          
            # Use gather to select the appropriate expert outputs
            selected_experts = expert_outputs.gather(1, expert_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, D)).squeeze(1)
            final_output += gate_weight * selected_experts
          
        return final_output.view(B, T, D)


# GraphAttentionLayer to support weighted adjacency
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        """
        Graph Attention Layer supporting weighted adjacency matrices.
        
        Args:
            in_features (int): Input feature dimension
            out_features (int): Output feature dimension
            dropout (float): Dropout rate (default: 0.1)
            alpha (float): LeakyReLU negative slope (default: 0.2)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
      
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.bias = nn.Parameter(torch.zeros(out_features))
      
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
      
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.layer_norm = nn.LayerNorm(out_features)
      
    def forward(self, h, adj):
        """
        Forward pass for GraphAttentionLayer.
        
        Args:
            h (torch.Tensor): Input node features of shape (batch_size, num_nodes, in_features)
            adj (torch.Tensor): Adjacency matrix of shape (batch_size, num_nodes, num_nodes)
        
        Returns:
            torch.Tensor: Output features of shape (batch_size, num_nodes, out_features)
        """
        Wh = torch.matmul(h, self.W)
        B, N, _ = Wh.shape
      
        # Self-attention mechanism
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = self.leakyrelu(Wh1 + Wh2.transpose(-1, -2))
      
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
        h_prime = torch.matmul(attention, Wh) + self.bias
        h_prime = self.layer_norm(h_prime)
      
        return h_prime


# Multi-Head Graph Attention
class MultiHeadGraphAttention(nn.Module):
    def __init__(self, in_features, out_features, num_heads=8, dropout=0.1):
        """
        Multi-Head Graph Attention mechanism.
        
        Args:
            in_features (int): Input feature dimension
            out_features (int): Output feature dimension (must be divisible by num_heads)
            num_heads (int): Number of attention heads (default: 8)
            dropout (float): Dropout rate (default: 0.1)
        """
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.head_dim = out_features // num_heads
      
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, self.head_dim, dropout)
            for _ in range(num_heads)
        ])
      
        self.out_proj = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)
      
    def forward(self, x, adj):
        """
        Forward pass for MultiHeadGraphAttention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, in_features)
            adj (torch.Tensor): Adjacency matrix of shape (batch_size, num_nodes, num_nodes)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_nodes, out_features)
        """
        head_outputs = []
        for attention in self.attentions:
            head_outputs.append(attention(x, adj))
          
        concat_output = torch.cat(head_outputs, dim=-1)
        output = self.out_proj(concat_output)
        output = self.dropout(output)
        output = self.layer_norm(output + x) # Residual connection
      
        return output


# Fusion Layer-II
class FusionLayer(nn.Module):
    def __init__(self, hidden_dim, num_modalities):
        """
        Fusion Layer combining graph attention, MoE, SwiGLU with residual connections and normalization.
        
        Args:
            hidden_dim (int): Hidden dimension (embedding dimension)
            num_modalities (int): Number of modalities (used in graph attention)
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
        self.residual_scale = nn.Parameter(torch.ones(3))
      
    def forward(self, x, adj):
        """
        Forward pass for FusionLayer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_modalities, hidden_dim)
            adj (torch.Tensor): Adjacency matrix of shape (batch_size, num_modalities, num_modalities)
        
        Returns:
            torch.Tensor: Output tensor of same shape as input
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
    def __init__(self, input_dim=emb_dim, hidden_dim=emb_dim, num_modalities=4, num_layers=4,
                 learnable_adj=False, dynamic_adj=False, directed=False, hierarchy=None):
        """
        Enhanced Fusion Module for multimodal data with graph attention, fusion layers, and output projection.
        
        Args:
            input_dim (int): Input embedding dimension (default: emb_dim)
            hidden_dim (int): Hidden dimension (default: emb_dim)
            num_modalities (int): Number of input modalities (default: 4)
            num_layers (int): Number of fusion layers (default: 4)
            learnable_adj (bool): Use learnable adjacency (default: False)
            dynamic_adj (bool): Compute dynamic adjacency from cosine similarity (default: False)
            directed (bool): Use directed graph for adjacency (default: False)
            hierarchy (torch.Tensor, optional): Hierarchy mask tensor of shape (num_modalities, num_modalities) (default: None)
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
                nn.Linear(input_dim, hidden_dim),
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
                   
        self.final_norm = RMSNorm(input_dim)
                   
        # For learnable adjacency (sigmoid-gated matrix)
        if self.learnable_adj:
            self.adj_logits = nn.Parameter(torch.zeros(num_modalities, num_modalities)) # Init to zeros for sigmoid ~0.5
          
    def create_adjacency_matrix(self, projected_embeddings):
        """
        Create adjacency matrix, potentially dynamic, learnable, with self-loops and hierarchy mask.
        
        Args:
            projected_embeddings (list of torch.Tensor): List of projected embeddings, each of shape (batch_size, hidden_dim)
        
        Returns:
            torch.Tensor: Adjacency matrix of shape (batch_size, num_modalities, num_modalities)
        """
        batch_size = projected_embeddings[0].shape[0]
        device = projected_embeddings[0].device
        M = self.num_modalities
      
        # Base adjacency
        if self.dynamic_adj:
            # Dynamic: Compute from cosine similarity
            emb_stack = torch.stack(projected_embeddings, dim=1) # [B, M, H]
            emb_norm = F.normalize(emb_stack, p=2, dim=-1)
            cos_sim = emb_norm @ emb_norm.transpose(1, 2) # [B, M, M]
            adj = (cos_sim + 1.0) / 2.0 # Normalize to [0, 1]
          
            if not self.directed:
                adj = (adj + adj.transpose(1, 2)) / 2.0 # Symmetrize unless directed
              
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
        # Example: If hierarchy is provided (e.g., a fixed M x M tensor with 1s for allowed edges in subgraphs/direct paths)
        if self.hierarchy is not None:
            # Expand to batch if needed
            if self.hierarchy.dim() == 2:
                hierarchy_mask = self.hierarchy.unsqueeze(0).expand(batch_size, -1, -1).to(device)
            else:
                hierarchy_mask = self.hierarchy.to(device)
            adj = adj * hierarchy_mask # Mask to enforce hierarchy (e.g., directed edges or subgraph connectivity)
          
        return adj
      
    def forward(self, modality_embeddings):
        """
        Forward pass for EnhancedFusionModule.
        
        Args:
            modality_embeddings (list of torch.Tensor): List of modality embeddings, each of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Fused output of shape (batch_size, input_dim)
        """
        batch_size = modality_embeddings[0].shape[0]
        device = modality_embeddings[0].device
      
        # Project inputs
        projected_embeddings = []
        for i, embedding in enumerate(modality_embeddings):
            projected = self.input_projections[i](embedding)
            projected_embeddings.append(projected)
          
        # Create adjacency (now dynamic/learnable)
        adj = self.create_adjacency_matrix(projected_embeddings)
      
        # Stack modalities
        x = torch.stack(projected_embeddings, dim=1)
      
        # Graph attention (Fusion Layer-I)
        attended_x = self.graph_attention(x, adj)
        x = x + attended_x # Residual
      
        # Multi-layer fusion (Fusion Layer-II)
        for layer in self.fusion_layers:
            x = layer(x, adj)
          
        # Global pooling (Fusion Layer-III)
        attention_weights = F.softmax(x.mean(dim=-1), dim=-1)
        fused_output = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
      
        # Final projection (Fusion Layer-III)
        output = self.output_projection(fused_output)
        output = self.final_norm(output)
      
        return output


# Customized T5 Time Series Encoder
class T5_TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim=3, config=Time_T5_config):
        """
        Customized T5 based encoder for time series data with projection, positional encoding, LoRA, and multi-scale pooling.
        
        Args:
            input_dim (int): Input channel dimension (default: 3 for x,y,z accelerometer data)
            config (T5Config): T5 model configuration (default: Time_T5_config)
        """
        super().__init__()
        self.config = config
      
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.d_model // 2, config.d_model)
        )
      
        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(ac_max_length, config.d_model))
      
        # Enhanced encoder with LoRA
        self.encoder = T5Stack(config, embed_tokens=None)
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=16, # Increased rank
            lora_alpha=32, # Increased alpha
            lora_dropout=0.1,
            target_modules=["q", "k", "v"] # More modules
        )
      
        self.encoder = get_peft_model(self.encoder, lora_config)
      
        # Multi-scale pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
      
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )
      
    def forward(self, x):
        """
        Forward pass for T5_TimeSeriesEncoder.
        
        Args:
            x (torch.Tensor): Input time series of shape (batch_size, sequence_length, input_dim)
        
        Returns:
            torch.Tensor: Encoded features of shape (batch_size, d_model)
        """
        B, T, C = x.shape
        x = self.input_projection(x)
      
        # Add positional encoding
        x = x + self.positional_encoding[:T].unsqueeze(0)
      
        # Attention mask
        attention_mask = torch.ones(B, T, device=x.device, dtype=torch.long)
      
        # Encode
        encoder_outputs = self.encoder(
            inputs_embeds=x,
            attention_mask=attention_mask
        )
      
        hidden_states = encoder_outputs.last_hidden_state
      
        # Multi-scale pooling
        avg_pooled = self.global_pool(hidden_states.transpose(1, 2)).squeeze(-1)
        max_pooled = self.max_pool(hidden_states.transpose(1, 2)).squeeze(-1)
      
        # Combine pooled features
        combined = torch.cat([avg_pooled, max_pooled], dim=-1)
        output = self.output_projection(combined)
      
        return output


# DART-CNN Encoder with dual attention and temporal processing
class DART_CNNEncoder(nn.Module):
    def __init__(self, input_shape, emb_dim=768):
        """
        DART-CNN Encoder with convolutions, dual attention (spatial and channel), FC layers, and temporal RNNs.
        
        Args:
            input_shape (tuple): Input frame shape (height, width), e.g., (12,16) or (32,16)
            emb_dim (int): Output embedding dimension (default: 768)
        """
        super(DART_CNNEncoder, self).__init__()
        height, width = input_shape
      
        # Convolutional layers with residual connections
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
      
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
      
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
                                 
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
      
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
      
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512 // 16),
            nn.ReLU(),
            nn.Linear(512 // 16, 512),
            nn.Sigmoid()
        )
      
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
      
        # Enhanced fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512, emb_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(emb_dim, emb_dim)
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
        Forward pass for DART_CNNEncoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, height, width)
        
        Returns:
            torch.Tensor: Encoded features of shape (batch_size, emb_dim)
        """
        B, T, H, W = x.shape
        x = x.view(B * T, 1, H, W)
      
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
      
        x = F.max_pool2d(x, 2)
      
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
      
        # Attention mechanisms
        spatial_att = self.spatial_attention(x)
        channel_att = self.channel_attention(x).view(B * T, 512, 1, 1)
        x = x * spatial_att * channel_att
      
        # Global pooling and FC
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
      
        # Reshape for temporal processing
        x = x.view(B, T, -1)
      
        # Temporal processing with LSTM, RNN, and GRU
        lstm_out, _ = self.lstm(x)
        rnn_out, _ = self.rnn(lstm_out)
        gru_out, _ = self.gru(rnn_out)
      
        # Combine with residual connection
        x = x + gru_out
      
        # Global temporal pooling
        return x.mean(dim=1)


# MHARFedLLM Activity Classification Model
class ActivityClassificationModel(nn.Module):
    def __init__(self, hidden_dim=emb_dim, num_classes=7):
        """
        Activity Classification Model with modality encoders, fusion, and classifier.
        
        Args:
            hidden_dim (int): Hidden/embedding dimension (default: emb_dim=512)
            num_classes (int): Number of output classes (default: 7)
        """
        super().__init__()
        # Modality specific encoders
        self.act_encoder = T5_TimeSeriesEncoder(input_dim=3)
        self.acw_encoder = T5_TimeSeriesEncoder(input_dim=3)
        self.dc_encoder = DART_CNNEncoder(input_shape=(12, 16), emb_dim=hidden_dim)
        self.pm_encoder = DART_CNNEncoder(input_shape=(32, 16), emb_dim=hidden_dim)
      
        # MAGNET fusion module
        self.fusion = EnhancedFusionModule(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_modalities=4,
            num_layers=3,
            learnable_adj=True,
            dynamic_adj=True,
            directed=False,  # Not used in the code and paper
            hierarchy=None   # Not used in the code and paper
        )
      
        # Multi Layer Hierarchical classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.4),
          
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
          
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(0.2),
          
            nn.Linear(hidden_dim // 4, num_classes)
        )
      
        # Learnable missing embedding
        self.missing_embedding = nn.Parameter(torch.randn(1, hidden_dim))
      
        # Modality-specific learnable weights
        self.modality_weights = nn.Parameter(torch.ones(4))
      
    def forward(self, act_tensor, acw_tensor, dc_tensor, pm_tensor):
        """
        Forward pass for ActivityClassificationModel.
        
        Args:
            act_tensor (torch.Tensor, optional): Accelerometer thigh data of shape (batch_size, ac_max_length=500, 3)
            acw_tensor (torch.Tensor, optional): Accelerometer wrist data of shape (batch_size, ac_max_length=500, 3)
            dc_tensor (torch.Tensor, optional): Depth camera data of shape (batch_size, dc_max_length=75, 12, 16)
            pm_tensor (torch.Tensor, optional): Pressure mat data of shape (batch_size, pm_max_length=75, 32, 16)
        
        Returns:
            tuple: (logits of shape (batch_size, num_classes), act_emb, acw_emb, dc_emb, pm_emb each of shape (batch_size, hidden_dim))
        """
        embeddings = []
        batch_size = (act_tensor.shape[0] if act_tensor is not None else
                      acw_tensor.shape[0] if acw_tensor is not None else
                      dc_tensor.shape[0] if dc_tensor is not None else
                      pm_tensor.shape[0])
      
        # Encode modalities
        if act_tensor is not None:
            act_emb = self.act_encoder(act_tensor)
        else:
            act_emb = self.missing_embedding.expand(batch_size, -1)
        embeddings.append(act_emb * self.modality_weights[0])
      
        if acw_tensor is not None:
            acw_emb = self.acw_encoder(acw_tensor)
        else:
            acw_emb = self.missing_embedding.expand(batch_size, -1)
        embeddings.append(acw_emb * self.modality_weights[1])
      
        if dc_tensor is not None:
            dc_emb = self.dc_encoder(dc_tensor)
        else:
            dc_emb = self.missing_embedding.expand(batch_size, -1)
        embeddings.append(dc_emb * self.modality_weights[2])
      
        if pm_tensor is not None:
            pm_emb = self.pm_encoder(pm_tensor)
        else:
            pm_emb = self.missing_embedding.expand(batch_size, -1)
        embeddings.append(pm_emb * self.modality_weights[3])
      
        # MAGNET fusion
        fused_features = self.fusion(embeddings)

        # Classifier
        logits = self.classifier(fused_features)
      
        return logits, act_emb, acw_emb, dc_emb, pm_emb
      
    def get_moe_load_balancing_loss(self):
        """Collect load balancing losses from all MoE layers"""
        total_loss = 0.0
        count = 0
      
        for module in self.modules():
            if isinstance(module, MixtureOfExperts):
                total_loss += module.load_balancing_loss
                count += 1
              
        return total_loss / count if count > 0 else 0.0


# Dataset
class SensorDataset(Dataset):
    def __init__(self, act_dir, acw_dir, dc_dir, pm_dir, participant_ids):
        """
        Dataset for loading and processing multimodal sensor data from directories.
        
        Args:
            act_dir (str): Directory for accelerometer thigh data
            acw_dir (str): Directory for accelerometer wrist data
            dc_dir (str): Directory for depth camera data
            pm_dir (str): Directory for pressure mat data
            participant_ids (list): List of participant IDs (strings like '01', '02', etc.)
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
                    label = activity_id - 1
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
        return len(self.data)
      
    def __getitem__(self, idx):
        return self.data[idx]


def load_sensor_file(filepath):
    """
    Load and process sensor data from CSV file, converting timestamps to Unix time.
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        numpy.ndarray: Loaded data array with timestamps in first column, features in others; or None on error
        Dimensions: (num_samples, num_columns) where num_columns depends on modality:
        - act/acw: 4 (timestamp, x, y, z)
        - dc: 193 (timestamp + 192 features)
        - pm: 513 (timestamp + 512 features)
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
    Apply Gaussian noise augmentation to data.
    
    Args:
        data (numpy.ndarray): Input data array
        
    Returns:
        numpy.ndarray: Augmented data of same shape
    """
    noise = np.random.normal(0, 0.01, data.shape)
    return data + noise


def normalize_data(data):
    """
    Normalize data by subtracting mean and dividing by std (z-score normalization).
    
    Args:
        data (numpy.ndarray): Input data array
        
    Returns:
        numpy.ndarray: Normalized data of same shape
    """
    data_mean = np.mean(data, axis=0, keepdims=True)
    data_std = np.std(data, axis=0, keepdims=True)
    return (data - data_mean) / (data_std + 1e-8)


def resample_data(data, target_length):
    """
    Resample data to a target length using linear interpolation or repetition.
    
    Args:
        data (numpy.ndarray): Input data, 2D (time, features) or 3D (time, height, width)
        target_length (int): Desired time dimension length
        
    Returns:
        numpy.ndarray: Resampled data with time dimension = target_length
    """
    if data is None or len(data) == 0:
        # Return zeros with correct shape
        if len(data.shape) == 2:
            return np.zeros((target_length, data.shape[1]))
        else:
            return np.zeros((target_length,) + data.shape[1:])
          
    if len(data) == target_length:
        return data
      
    if len(data) == 1:
        # If only one sample, repeat it
        return np.repeat(data, target_length, axis=0)
      
    # Use interpolation for resampling
    x = np.linspace(0, 1, len(data))
    x_new = np.linspace(0, 1, target_length)
  
    if len(data.shape) == 2:
        # 2D data (time, features)
        interpolated = np.zeros((target_length, data.shape[1]))
      
        for i in range(data.shape[1]):
            interpolator = interp1d(x, data[:, i], kind='linear', fill_value='extrapolate')
            interpolated[:, i] = interpolator(x_new)
          
        return interpolated
      
    else:
        # 3D data (time, height, width)
        interpolated = np.zeros((target_length,) + data.shape[1:])
      
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                interpolator = interp1d(x, data[:, i, j], kind='linear', fill_value='extrapolate')
                interpolated[:, i, j] = interpolator(x_new)
              
        return interpolated


def split_windows(act_data, acw_data, dc_data, pm_data):
    """
    Split multimodal data into overlapping time windows.
    
    Args:
        act_data (numpy.ndarray): Accelerometer thigh data (num_samples, 4) [timestamp, x, y, z]
        acw_data (numpy.ndarray): Accelerometer wrist data (num_samples, 4) [timestamp, x, y, z]
        dc_data (numpy.ndarray): Depth camera data (num_samples, 193) [timestamp + 192 features]
        pm_data (numpy.ndarray): Pressure mat data (num_samples, 513) [timestamp + 512 features]
        
    Returns:
        list: List of window tuples [(act_window, acw_window, dc_window, pm_window), ...]
              where each window is numpy.ndarray or None, with shapes:
              - act/acw: (ac_max_length=500, 3)
              - dc: (dc_max_length=75, 12, 16)
              - pm: (pm_max_length=75, 32, 16)
    """
    # Handle case where all data is None
    if all(data is None for data in [act_data, acw_data, dc_data, pm_data]):
        return []
      
    # Get valid time ranges
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
                act_window = act_data[act_mask, 1:4]
              
        acw_window = None
        if acw_data is not None and len(acw_data) > 0:
            acw_mask = (acw_data[:, 0] >= current_time) & (acw_data[:, 0] < window_end)
            if acw_mask.sum() > 0:
                acw_window = acw_data[acw_mask, 1:4]
              
        dc_window = None
        if dc_data is not None and len(dc_data) > 0:
            dc_mask = (dc_data[:, 0] >= current_time) & (dc_data[:, 0] < window_end)
            if dc_mask.sum() > 0:
                dc_window_flat = dc_data[dc_mask, 1:]
                if dc_window_flat.shape[1] == 192: # 12 * 16 = 192
                    dc_window = dc_window_flat.reshape(-1, 12, 16)
                  
        pm_window = None
        if pm_data is not None and len(pm_data) > 0:
            pm_mask = (pm_data[:, 0] >= current_time) & (pm_data[:, 0] < window_end)
            if pm_mask.sum() > 0:
                pm_window_flat = pm_data[pm_mask, 1:]
                if pm_window_flat.shape[1] == 512: # 32 * 16 = 512
                    pm_window = pm_window_flat.reshape(-1, 32, 16)
                  
        # Skip if no valid data in this window
        if all(w is None for w in [act_window, acw_window, dc_window, pm_window]):
            current_time += increment
            continue
          
        # Process windows
        if act_window is not None and len(act_window) > 0:
            act_window = resample_data(act_window, ac_max_length)
            act_window = augment_data(normalize_data(act_window))
          
        if acw_window is not None and len(acw_window) > 0:
            acw_window = resample_data(acw_window, ac_max_length)
            acw_window = augment_data(normalize_data(acw_window))
          
        if dc_window is not None and len(dc_window) > 0:
            dc_window = resample_data(dc_window, dc_max_length)
            dc_window = normalize_data(dc_window.reshape(-1, 12*16)).reshape(-1, 12, 16)
          
        if pm_window is not None and len(pm_window) > 0:
            pm_window = resample_data(pm_window, pm_max_length)
            pm_window = normalize_data(pm_window.reshape(-1, 32*16)).reshape(-1, 32, 16)
          
        windows.append([act_window, acw_window, dc_window, pm_window])
        current_time += increment
      
    return windows


def collate_fn(batch):
    """
    Collate function for batching multimodal data, handling None tensors.
    
    Args:
        batch (list): List of samples [(act_tensor, acw_tensor, dc_tensor, pm_tensor, label), ...]
        
    Returns:
        tuple: (act_batch, acw_batch, dc_batch, pm_batch, label_batch)
               where each batch is torch.Tensor or None, shapes:
               - act/acw_batch: (batch_size, ac_max_length=500, 3)
               - dc_batch: (batch_size, dc_max_length=75, 12, 16)
               - pm_batch: (batch_size, pm_max_length=75, 32, 16)
               - label_batch: (batch_size,)
    """
    act_list, acw_list, dc_list, pm_list, labels = [], [], [], [], []
  
    for act, acw, dc, pm, y in batch:
        act_list.append(act)
        acw_list.append(acw)
        dc_list.append(dc)
        pm_list.append(pm)
        labels.append(y)
      
    # Create batches, handling None values
    act_batch = None
    acw_batch = None
    dc_batch = None
    pm_batch = None
  
    # Stack non-None tensors
    valid_acts = [x for x in act_list if x is not None]
  
    if valid_acts:
        act_batch = torch.stack(valid_acts)
    valid_acws = [x for x in acw_list if x is not None]
  
    if valid_acws:
        acw_batch = torch.stack(valid_acws)
    valid_dcs = [x for x in dc_list if x is not None]
  
    if valid_dcs:
        dc_batch = torch.stack(valid_dcs)
    valid_pms = [x for x in pm_list if x is not None]
  
    if valid_pms:
        pm_batch = torch.stack(valid_pms)
    label_batch = torch.tensor(labels, dtype=torch.long)
  
    # Check if we have at least one valid modality
    if all(batch is None for batch in [act_batch, acw_batch, dc_batch, pm_batch]):
        raise ValueError("All modalities are None for the batch")
      
    return act_batch, acw_batch, dc_batch, pm_batch, label_batch


# Input directories
act_dir = '/content/data/act'
acw_dir = '/content/data/acw'
dc_dir = '/content/data/dc_0.05_0.05'
pm_dir = '/content/data/pm_1.0_1.0'

# Create datasets and dataloaders
all_participants = [f'{i:02d}' for i in range(1, 31)]
random.shuffle(all_participants)

train_participants = all_participants[:21]
val_participants = all_participants[21:24]
test_participants = all_participants[24:]

print("Creating datasets...")
train_dataset = SensorDataset(act_dir, acw_dir, dc_dir, pm_dir, train_participants)
val_dataset = SensorDataset(act_dir, acw_dir, dc_dir, pm_dir, val_participants)
test_dataset = SensorDataset(act_dir, acw_dir, dc_dir, pm_dir, test_participants)
print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# Instantiate the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Training step with MoE loss
def training_step_with_moe_loss(model, act_x, acw_x, dc_x, pm_x, y, criterion):
    """
    Compute loss including classification and MoE load balancing.
    
    Args:
        model (nn.Module): The activity classification model
        act_x (torch.Tensor or None): Accelerometer thigh batch
        acw_x (torch.Tensor or None): Accelerometer wrist batch
        dc_x (torch.Tensor or None): Depth camera batch
        pm_x (torch.Tensor or None): Pressure mat batch
        y (torch.Tensor): Labels of shape (batch_size,)
        criterion (nn.Module): Cross-entropy loss function
        
    Returns:
        tuple: (total_loss, logits, classification_loss, load_balancing_loss)
    """
    logits, _, _, _, _ = model(act_x, acw_x, dc_x, pm_x)
    classification_loss = criterion(logits, y)
  
    # Collect load balancing losses from all MoE layers
    load_balancing_loss = model.get_moe_load_balancing_loss()
    total_loss = classification_loss + load_balancing_loss
  
    return total_loss, logits, classification_loss, load_balancing_loss


# Modified evaluate function to include precision, recall, and classification report
def evaluate_with_moe_loss(model, loader, device):
    """
    Evaluate model on a dataset, computing metrics and collecting embeddings.
    
    Args:
        model (nn.Module): The activity classification model
        loader (DataLoader): DataLoader for evaluation
        device (torch.device): Device to run evaluation on
        
    Returns:
        tuple: (accuracy, weighted_f1, precision, recall, act_embeddings, acw_embeddings, dc_embeddings, pm_embeddings, all_labels, all_probs, cm)
               Embeddings are numpy arrays of shape (num_samples, hidden_dim)
    """
    model.eval()
  
    total, correct = 0, 0
    all_preds, all_labels = [], []
    all_probs = [] # For ROC and precision-recall curves
    act_embeddings, acw_embeddings, dc_embeddings, pm_embeddings = [], [], [], []
    total_moe_loss = 0.0
    num_batches = 0
  
    with torch.no_grad():
        for batch_idx, (act_x, acw_x, dc_x, pm_x, y) in enumerate(tqdm(loader, desc="Evaluating Test Set")):
            act_x = act_x.to(device) if act_x is not None else None
            acw_x = acw_x.to(device) if acw_x is not None else None
            dc_x = dc_x.to(device) if dc_x is not None else None
            pm_x = pm_x.to(device) if pm_x is not None else None
          
            y = y.to(device)
          
            logits, act_emb, acw_emb, dc_emb, pm_emb = model(act_x, acw_x, dc_x, pm_x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
          
            # Track MoE loss
            moe_loss = model.get_moe_load_balancing_loss()
            total_moe_loss += moe_loss if isinstance(moe_loss, (int, float)) else moe_loss.item()
          
            num_batches += 1
            correct += (preds == y).sum().item()
            total += y.size(0)
          
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
          
            act_embeddings.extend(act_emb.cpu().numpy())
            acw_embeddings.extend(acw_emb.cpu().numpy())
            dc_embeddings.extend(dc_emb.cpu().numpy())
            pm_embeddings.extend(pm_emb.cpu().numpy())
          
    accuracy = correct / total
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    cm = confusion_matrix(all_labels, all_preds)
    avg_moe_loss = total_moe_loss / num_batches
  
    # Generate and save classification report
    class_report = classification_report(all_labels, all_preds, digits=3)
    with open('/content/classification_report.txt', 'w') as f:
        f.write(class_report)
    print("\nClassification Report saved to '/content/classification_report.txt'")
  
    print("Classification Report:\n", class_report)
    print("Confusion Matrix:\n", cm)
    print("Per-Class F1 Scores:", per_class_f1)
    print(f"Average MoE Load Balancing Loss: {avg_moe_loss:.4f}")
  
    return accuracy, weighted_f1, precision, recall, np.array(act_embeddings), np.array(acw_embeddings), np.array(dc_embeddings), np.array(pm_embeddings), np.array(all_labels), np.array(all_probs), cm


# Modified training function to track only training and validation accuracies
def train_model_with_metrics():
    """
    Train the model with gradient accumulation, mixed precision, and early stopping.
    
    Returns:
        tuple: (model, train_losses, val_losses, train_accuracies, val_accuracies)
    """
    model = ActivityClassificationModel(hidden_dim=emb_dim, num_classes=7).to(device)
  
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
  
    class_counts = np.zeros(num_classes)
    labels = [y for _, _, _, _, y in train_dataset]
    unique_labels, counts = np.unique(labels, return_counts=True)
  
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
  
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6, verbose=True)
    scaler = GradScaler()
  
    best_val_acc = 0.0
    patience = 6
    patience_counter = 0
    max_epochs = 10
  
    # Track metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
  
    print("\nStarting training with accuracy metrics tracking...")
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        running_classification_loss = 0.0
        running_moe_loss = 0.0
        num_batches = 0
        train_correct = 0
        train_total = 0
      
        optimizer.zero_grad()
      
        tqdm_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", leave=False)
        for batch_idx, (act_x, acw_x, dc_x, pm_x, y) in enumerate(tqdm_loader):
            act_x = act_x.to(device) if act_x is not None else None
            acw_x = acw_x.to(device) if acw_x is not None else None
            dc_x = dc_x.to(device) if dc_x is not None else None
            pm_x = pm_x.to(device) if pm_x is not None else None
          
            y = y.to(device)
            with autocast(device_type='cuda'):
                total_loss, logits, classification_loss, moe_loss = training_step_with_moe_loss(
                    model, act_x, acw_x, dc_x, pm_x, y, criterion
                )
                total_loss = total_loss / accumulation_steps
              
            scaler.scale(total_loss).backward()
          
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
              
            # Track training accuracy
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)
            running_loss += total_loss.item() * accumulation_steps
            running_classification_loss += classification_loss.item()
            running_moe_loss += moe_loss if isinstance(moe_loss, (int, float)) else moe_loss.item()
            num_batches += 1
          
            tqdm_loader.set_postfix({
                'Total Loss': f'{running_loss/num_batches:.4f}',
                'Class Loss': f'{running_classification_loss/num_batches:.4f}',
                'MoE Loss': f'{running_moe_loss/num_batches:.4f}'
            })
          
        avg_total_loss = running_loss / num_batches
        train_acc = train_correct / train_total
      
        # Evaluate validation set
        model.eval()
      
        val_total, val_correct = 0, 0
        val_loss = 0.0
        num_val_batches = 0
      
        with torch.no_grad():
            for act_x, acw_x, dc_x, pm_x, y in val_loader:
                act_x = act_x.to(device) if act_x is not None else None
                acw_x = acw_x.to(device) if acw_x is not None else None
                dc_x = dc_x.to(device) if dc_x is not None else None
                pm_x = pm_x.to(device) if pm_x is not None else None
                y = y.to(device)
              
                with autocast(device_type='cuda'):
                    total_loss, logits, _, _ = training_step_with_moe_loss(model, act_x, acw_x, dc_x, pm_x, y, criterion)
                    preds = torch.argmax(logits, dim=1)
                  
                val_loss += total_loss.item()
                num_val_batches += 1
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
              
        avg_val_loss = val_loss / num_val_batches
        val_acc = val_correct / val_total
      
        # Store metrics
        train_losses.append(avg_total_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
      
        print(f"Epoch {epoch+1}/{max_epochs}:")
        print(f" Total Loss: {avg_total_loss:.4f}")
        print(f" Validation Loss: {avg_val_loss:.4f}")
        print(f" Train Acc: {train_acc:.3f}")
        print(f" Val Acc: {val_acc:.3f}")
      
        scheduler.step(val_acc)
      
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), '/content/best_model.pth')
            print(f" ✅ New best validation accuracy: {best_val_acc:.3f}")
        else:
            patience_counter += 1
            print(f" No improvement. Patience: {patience_counter}/{patience}")
          
        if patience_counter >= patience:
            print(f"\n⏰ Early stopping triggered after {epoch+1} epochs.")
            break
          
    return model, train_losses, val_losses, train_accuracies, val_accuracies


# Visualization function with confusion matrix
def create_visualizations(train_losses, val_losses, train_accuracies, val_accuracies, act_embeddings, acw_embeddings, dc_embeddings, pm_embeddings, all_labels, all_probs, cm):
    """
    Generate and save various visualization plots for training metrics and test embeddings.
    
    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
        train_accuracies (list): List of training accuracies per epoch
        val_accuracies (list): List of validation accuracies per epoch
        act_embeddings (numpy.ndarray): Accelerometer thigh embeddings (num_test_samples, hidden_dim)
        acw_embeddings (numpy.ndarray): Accelerometer wrist embeddings (num_test_samples, hidden_dim)
        dc_embeddings (numpy.ndarray): Depth camera embeddings (num_test_samples, hidden_dim)
        pm_embeddings (numpy.ndarray): Pressure mat embeddings (num_test_samples, hidden_dim)
        all_labels (numpy.ndarray): Test labels (num_test_samples,)
        all_probs (numpy.ndarray): Test probabilities (num_test_samples, num_classes)
        cm (numpy.ndarray): Confusion matrix (num_classes, num_classes)
    """
    # Set style
    sns.set_palette("tab10")
  
    # 1. Training and Validation Loss Curve
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
  
    # 2. Training and Validation Accuracy Curve
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
  
    # 3. t-SNE Visualization (Test set only)
    fused_embeddings = (act_embeddings + acw_embeddings + dc_embeddings + pm_embeddings) / 4
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_embeddings = tsne.fit_transform(fused_embeddings)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=all_labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Activity Class')
    plt.title('t-SNE Visualization of Test Set Fused Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(False)
    plt.savefig('/content/tsne_plot.png')
    plt.close()
  
    # 4. UMAP Visualization (Test set only)
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    umap_embeddings = umap_reducer.fit_transform(fused_embeddings)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=all_labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Activity Class')
    plt.title('UMAP Visualization of Test Set Fused Embeddings')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.grid(False)
    plt.savefig('/content/umap_plot.png')
    plt.close()
  
    # 5. ROC-AUC Curve (Test set)
    plt.figure(figsize=(10, 6))
    all_labels_onehot = F.one_hot(torch.tensor(all_labels), num_classes=7).numpy()
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
  
    # 6. Precision-Recall Curve (Test set)
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
  
    # 7. Confusion Matrix Visualization (Test set)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Test Set)')
    plt.savefig('/content/confusion_matrix.png')
    plt.close()
  
    print("\n✅ Visualizations saved successfully!")


# Run the enhanced training and evaluation
if __name__ == "__main__":
    # Train the model and collect accuracy metrics
    trained_model, train_losses, val_losses, train_accuracies, val_accuracies = train_model_with_metrics()
  
    # Final evaluation on test set
    print("\n" + "="*50)
    print("FINAL EVALUATION (TEST SET)")
    print("="*50)
  
    try:
        trained_model.load_state_dict(torch.load('/content/best_model.pth'))
    except FileNotFoundError:
        print("Best model file not found. Using final model state.")
    test_acc, test_f1, test_precision, test_recall, act_embeddings, acw_embeddings, dc_embeddings, pm_embeddings, test_labels, test_probs, test_cm = evaluate_with_moe_loss(
        trained_model, test_loader, device
    )
  
    print(f"\n✅ FINAL RESULTS (TEST SET):")
    print(f"Test Accuracy: {test_acc:.3f}")
    print(f"Test F1 Score: {test_f1:.3f}")
    print(f"Test Precision: {test_precision:.3f}")
    print(f"Test Recall: {test_recall:.3f}")
  
    print(f"Accelerometer Thigh Embeddings Shape: {act_embeddings.shape}")
    print(f"Accelerometer Wrist Embeddings Shape: {acw_embeddings.shape}")
    print(f"Depth Camera Embeddings Shape: {dc_embeddings.shape}")
    print(f"Pressure Mat Embeddings Shape: {pm_embeddings.shape}")
  
    # Save embeddings
    np.save('/content/act_embeddings.npy', act_embeddings)
    np.save('/content/acw_embeddings.npy', acw_embeddings)
    np.save('/content/dc_embeddings.npy', dc_embeddings)
    np.save('/content/pm_embeddings.npy', pm_embeddings)
  
    # Create visualizations for test set
    create_visualizations(
        train_losses, val_losses, train_accuracies, val_accuracies,
        act_embeddings, acw_embeddings, dc_embeddings, pm_embeddings, test_labels, test_probs, test_cm
    )
  
    print("\n✅ All tasks completed successfully!")
