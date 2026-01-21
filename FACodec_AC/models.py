import torch
import torch.nn as nn
from FACodec_AC.utils import snap_latent, QuantizerNames

class ConvFeedForward(nn.Module):
    """
    A feed-forward block with 1D convolution (kernel_size=3)
    to simulate a "filter size 2048" notion.
    """
    def __init__(self, d_model: int = 1024, d_ff: int = 2048, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size, padding=(kernel_size // 2))
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size, padding=(kernel_size // 2))
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x (Tensor [batch_size, d_model, seq_len])
        
        Returns:
            Tensor [batch_size, d_model, seq_len]
        """
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        return self.dropout(out)

class CondLayerNorm(nn.Module):
    """
    Normalizes input tensor x and applies affine modulation using parameters derived from a conditioning tensor.

    Args:
        d_model (int): Dimensionality of input features.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x (torch.Tensor): Tensor with shape [B, T, D] to be normalized.
            cond (torch.Tensor): Tensor with shape [B, T, 2D], split into gamma and beta for modulation.

        Returns:
            torch.Tensor: Normalized and modulated tensor with shape [B, T, D].
        """
        gamma, beta = cond.chunk(2, dim=-1)
        x_norm = self.norm(x)
        return x_norm * (1 + gamma) + beta

class CustomTransformerEncoderLayer(nn.Module):
    """
    A custom Transformer encoder layer with ConvFeedForward and conditional LayerNorm.
    """
    def __init__(self, d_model: int=1024, nhead: int=8, d_ff: int=2048, dropout: float=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        # conditional LayerNorm for post-FFN
        self.norm2 = CondLayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.conv_ff = ConvFeedForward(d_model=d_model, d_ff=d_ff, kernel_size=3, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        src_key_padding_mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        """
        Parameters:
            x (torch.Tensor): Input tensor of shape [B, T, D]
            cond (torch.Tensor): Conditioning tensor used for the conditional layer normalization (FiLM parameters).
            src_key_padding_mask (torch.BoolTensor, optional): Boolean mask indicating positions to ignore in the attention mechanism. Shape should conform to [B, T].

        Returns:
            torch.Tensor: Output tensor of shape [B, T, D] after applying self-attention, feed-forward operations, and conditional normalization.
        """
        # Self-attention block
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # Conv feed-forward block
        x_t = x.transpose(1, 2)  # [B, D, T]
        ff_out = self.conv_ff(x_t)
        ff_out = ff_out.transpose(1, 2)  # [B, T, D]
        x = x + self.dropout(ff_out)

        # Conditional LayerNorm with FiLM parameters
        return self.norm2(x, cond)

class CustomTransformerEncoder(nn.Module):
    """
    Stacks multiple Conditional Transformer encoder layers.
    """
    def __init__(self, num_layers: int=12, d_model: int=1024, nhead: int=8, d_ff: int=2048, dropout: float=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        src_key_padding_mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        # cond: [B, T, 2D]
        for layer in self.layers:
            x = layer(x, cond, src_key_padding_mask=src_key_padding_mask)
        return x

class DenoisingTransformerModel(nn.Module):
    def __init__(
        self,
        d_model: int = 1024,
        nhead: int = 8,
        num_layers: int = 12,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 807,
        FACodec_dim: int = 8,
        phone_vocab_size: int = 392,
        num_steps: int = 50,               # ← number of diffusion steps
    ):
        super().__init__()
        self.d_model = d_model
        self.num_steps = num_steps

        # embeddings & encoder
        self.proj_to_d_model = nn.Linear(FACodec_dim, d_model)
        self.pos_embedding   = nn.Embedding(max_seq_len, d_model)
        self.phone_embedding = nn.Embedding(phone_vocab_size + 1, d_model)
        self.encoder         = CustomTransformerEncoder(num_layers, d_model, nhead, d_ff, dropout)

        # ε-prediction head
        self.fc_out = nn.Linear(d_model, FACodec_dim)
        # zc2 head (unchanged)
        self.fc_zc2 = nn.Sequential(
            nn.Linear(d_model + FACodec_dim, 4 * FACodec_dim),
            nn.GELU(),
            nn.Linear(4 * FACodec_dim, FACodec_dim),
        )

        # FiLM conditioning
        self.phone_proj = nn.Linear(d_model, d_model * 2)
        self.t_embed    = nn.Embedding(num_steps, d_model * 2)
        self.dropout    = nn.Dropout(dropout)

        # --- Build DDPM schedule using linear betas as in DDPM paper ---
        betas = torch.linspace(1e-4, 0.02, num_steps, dtype=torch.float)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_abar", torch.sqrt(abar))         # [num_steps]
        self.register_buffer("sqrt_1mabar", torch.sqrt(1.0 - abar))   # [num_steps]

    def forward(
        self,
        zc1_noisy: torch.Tensor,        # [B, C, T]
        padded_phone_ids: torch.LongTensor,  # [B, T]
        t: torch.LongTensor,             # [B]
        padding_mask: torch.BoolTensor,  # [B, T]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [B,C,T] → [B,T,C]
        z = zc1_noisy.transpose(1, 2)
        bsz, seq_len, _ = z.shape

        # token + pos + phone embeddings
        pos_ids    = torch.arange(seq_len, device=z.device).unsqueeze(0)
        tok_emb    = self.proj_to_d_model(z)
        pos_emb    = self.pos_embedding(pos_ids)
        phone_emb  = self.phone_embedding(padded_phone_ids)
        phone_emb  = self.dropout(phone_emb)
        h = tok_emb + pos_emb + phone_emb

        # FiLM-style conditioning
        phone_cond = self.phone_proj(phone_emb)           # [B,T,2D]
        t_cond     = self.t_embed(t)                      # [B,2D]
        t_cond     = t_cond.unsqueeze(1).expand(-1, seq_len, -1)
        cond       = self.dropout(phone_cond) + t_cond

        # encode
        h = self.encoder(h, cond, src_key_padding_mask=padding_mask)

        # ε-prediction
        eps_pred = self.fc_out(h)                         # [B,T,C]

        # zc2 branch: use the *denoised* x₀ estimate
        # x0_hat = (z - √(1-ᾱ_t)*ε) / √ᾱ_t
        sa  = self.sqrt_abar[t].view(bsz, 1, 1)
        s1a = self.sqrt_1mabar[t].view(bsz, 1, 1)
        x0_hat = (z - s1a * eps_pred) / sa

        zc2_input = torch.cat([h, x0_hat.detach()], dim=-1)
        zc2_pred  = self.fc_zc2(zc2_input)

        # return shapes [B, C, T]
        return eps_pred.transpose(1, 2), zc2_pred.transpose(1, 2)



