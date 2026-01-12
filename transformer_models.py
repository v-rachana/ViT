"""
Vision Transformer Model Components
====================================
This module implements the core components of a Vision Transformer (ViT) architecture
for image classification tasks. The implementation follows the Vision Transformer paper
with modifications for CPU-friendly training (MWE-CPU).
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Converts input images into patch embeddings with positional encoding.
    
    This module divides an image into non-overlapping patches, projects each patch
    into an embedding space, and adds learnable positional embeddings to preserve
    spatial information. A classification (CLS) token is prepended to the sequence.
    
    Args:
        in_channels (int): Number of input image channels (1 for grayscale, 3 for RGB).
        patch_size (int): Spatial dimension of each patch (patches are square).
        emb_size (int): Dimensionality of the embedding space.
        img_size (int): Spatial dimension of input images (assumed square).
    
    Attributes:
        patch_size (int): Size of each image patch.
        projection (nn.Conv2d): Convolutional layer for patch projection.
        cls_token (nn.Parameter): Learnable classification token.
        positional_embedding (nn.Parameter): Learnable positional embeddings.
    """
    def __init__(self, in_channels=3, patch_size=4, emb_size=64, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        
        # Convolutional projection: each patch becomes an embedding vector
        self.projection = nn.Conv2d(
            in_channels, 
            emb_size, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Classification token (prepended to patch sequence)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        
        # Positional embeddings: one for CLS token + one for each patch
        self.positional_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, emb_size)
        )

    def forward(self, x):
        """
        Forward pass: convert image to patch embeddings.
        
        Args:
            x (torch.Tensor): Input images of shape [batch, channels, height, width].
        
        Returns:
            torch.Tensor: Patch embeddings with CLS token, shape [batch, num_patches+1, emb_size].
        """
        # Project patches: [B, C, H, W] -> [B, emb_size, H/patch, W/patch]
        x = self.projection(x)
        
        # Flatten spatial dimensions: [B, emb_size, H/patch, W/patch] -> [B, emb_size, num_patches]
        x = x.flatten(2)
        
        # Transpose: [B, emb_size, num_patches] -> [B, num_patches, emb_size]
        x = x.transpose(1, 2)
        
        # Add CLS token to each sample in batch
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embeddings
        x = x + self.positional_embedding
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism for Vision Transformers.
    
    Implements scaled dot-product attention across multiple parallel attention heads,
    allowing the model to attend to different types of information simultaneously.
    Attention weights are stored for visualization purposes.
    
    Args:
        emb_size (int): Embedding dimension (must be divisible by num_heads).
        num_heads (int): Number of parallel attention heads.
        dropout (float): Dropout probability for attention weights.
    
    Attributes:
        attention_weights (torch.Tensor): Cached attention weights from last forward pass.
    """
    def __init__(self, emb_size, num_heads=8, dropout=0.1):
        super().__init__()
        assert emb_size % num_heads == 0, "emb_size must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.head_dim = emb_size // num_heads
        
        # Single linear layer for Q, K, V projections (more efficient)
        self.qkv_projection = nn.Linear(emb_size, emb_size * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(emb_size, emb_size)
        
        # Store attention weights for visualization
        self.attention_weights = None

    def forward(self, x):
        """
        Forward pass: compute multi-head self-attention.
        
        Args:
            x (torch.Tensor): Input sequence of shape [batch, num_tokens, emb_size].
        
        Returns:
            torch.Tensor: Attention output of shape [batch, num_tokens, emb_size].
        """
        batch_size, num_tokens, emb_size = x.shape
        
        # Compute Q, K, V in one pass: [B, T, emb_size] -> [B, T, 3*emb_size]
        qkv = self.qkv_projection(x)
        
        # Reshape and split: [B, T, 3*emb_size] -> [B, T, 3, num_heads, head_dim]
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        
        # Permute to separate Q, K, V: [3, B, num_heads, T, head_dim]
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        # Scaled dot-product attention: Q @ K^T / sqrt(head_dim)
        scale_factor = self.emb_size ** -0.5
        attention_scores = (q @ k.transpose(-2, -1)) * scale_factor
        attention_probs = attention_scores.softmax(dim=-1)
        
        # Store attention weights for visualization
        self.attention_weights = attention_probs
        
        # Apply dropout and compute weighted values
        attention_probs = self.attn_dropout(attention_probs)
        attended_values = attention_probs @ v  # [B, num_heads, T, head_dim]
        
        # Concatenate heads: [B, num_heads, T, head_dim] -> [B, T, emb_size]
        attended_values = attended_values.transpose(1, 2).reshape(
            batch_size, num_tokens, emb_size
        )
        
        # Final output projection
        return self.output_projection(attended_values)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with pre-norm architecture.
    
    Each encoder layer consists of:
    1. Multi-head self-attention with residual connection
    2. Feed-forward network with residual connection
    
    Uses pre-norm architecture (LayerNorm before sub-layers) for stable training.
    
    Args:
        emb_size (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        forward_expansion (int): Expansion factor for feed-forward network.
        dropout (float): Dropout probability.
    """
    def __init__(self, emb_size, num_heads, forward_expansion, dropout):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(emb_size)
        self.multi_head_attention = MultiHeadAttention(emb_size, num_heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        
        self.layernorm2 = nn.LayerNorm(emb_size)
        
        # Feed-forward network with expansion
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, forward_expansion * emb_size),
            nn.GELU(),  # GELU activation (used in ViT)
            nn.Linear(forward_expansion * emb_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Forward pass: apply attention and feed-forward with residual connections.
        
        Args:
            x (torch.Tensor): Input sequence [batch, num_tokens, emb_size].
        
        Returns:
            torch.Tensor: Output sequence [batch, num_tokens, emb_size].
        """
        # Pre-norm attention with residual connection
        normalized_x = self.layernorm1(x)
        attention_output = self.multi_head_attention(normalized_x)
        x = x + self.attention_dropout(attention_output)
        
        # Pre-norm feed-forward with residual connection
        normalized_x = self.layernorm2(x)
        ff_output = self.feed_forward(normalized_x)
        x = x + ff_output
        
        return x


class VisionTransformer(nn.Module):
    """
    Complete Vision Transformer (ViT) model for image classification.
    
    The Vision Transformer processes images by:
    1. Dividing images into patches and embedding them
    2. Processing patch sequences through transformer encoder layers
    3. Using the CLS token for final classification
    
    Args:
        in_channels (int): Number of input image channels.
        patch_size (int): Size of image patches (patches are square).
        emb_size (int): Embedding dimension.
        img_size (int): Input image size (assumed square).
        num_classes (int): Number of output classes.
        depth (int): Number of transformer encoder layers.
        num_heads (int): Number of attention heads.
        forward_expansion (int): FFN expansion factor.
        dropout (float): Dropout probability.
    """
    def __init__(self, in_channels=3, patch_size=4, emb_size=64, img_size=32, 
                 num_classes=10, depth=7, num_heads=8, forward_expansion=4, dropout=0.0):
        super().__init__()
        
        # Patch embedding module
        self.patch_embedding = PatchEmbedding(
            in_channels, patch_size, emb_size, img_size
        )
        
        # Stack of transformer encoder layers
        encoder_layers = [
            TransformerEncoderLayer(emb_size, num_heads, forward_expansion, dropout)
            for _ in range(depth)
        ]
        self.transformer_encoders = nn.Sequential(*encoder_layers)
        
        # Final layer normalization and classification head
        self.final_layernorm = nn.LayerNorm(emb_size)
        self.classification_head = nn.Linear(emb_size, num_classes)

    def forward(self, x):
        """
        Forward pass: classify input images.
        
        Args:
            x (torch.Tensor): Input images [batch, channels, height, width].
        
        Returns:
            torch.Tensor: Class logits [batch, num_classes].
        """
        # Convert image to patch embeddings with CLS token
        x = self.patch_embedding(x)  # [B, num_patches+1, emb_size]
        
        # Process through transformer encoders
        x = self.transformer_encoders(x)  # [B, num_patches+1, emb_size]
        
        # Extract CLS token (first token) and normalize
        cls_token = x[:, 0]  # [B, emb_size]
        cls_token = self.final_layernorm(cls_token)
        
        # Classification head
        logits = self.classification_head(cls_token)  # [B, num_classes]
        
        return logits