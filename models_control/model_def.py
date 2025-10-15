import torch
import torch.nn as nn
from typing import Tuple


class CausalPedestrianPredictor(nn.Module):
    """
    Causal sequence-to-sequence model for pedestrian positions with control inputs.

    Inputs:
      - u: control sequence, shape [batch, T, u_dim]
      - p_veh0: initial vehicle position, shape [batch, 2]
      - p_ped0: initial pedestrian position, shape [batch, 2]

    Outputs:
      - p_ped_seq: predicted pedestrian positions for steps 1..T, shape [batch, T, 2]

    Causality:
      Each p_ped_t depends only on (u_0, ..., u_t) and initial states via
      a recurrent GRU, where the initial hidden state is a learned function of
      (p_veh0, p_ped0). The step input depends only on u_t.
    """

    def __init__(
        self,
        u_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        self.u_dim = u_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Project control input per time step
        self.input_proj = nn.Sequential(
            nn.Linear(u_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Initialize recurrent core
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Optional normalization on hidden before output
        self.norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()

        # Map hidden state to delta position (residual over last ped position)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )

        # Initialize hidden state from initial conditions (p_veh0, p_ped0)
        self.h0_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        u: torch.Tensor,
        p_veh0: torch.Tensor,
        p_ped0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            u: [B, T, u_dim]
            p_veh0: [B, 2]
            p_ped0: [B, 2]

        Returns:
            p_ped_seq: [B, T, 2] positions for steps 1..T
        """
        B, T, _ = u.shape

        # Initial hidden from initial positions only
        h0_single_layer = self.h0_mlp(torch.cat([p_veh0, p_ped0], dim=-1))  # [B, H]
        # Repeat for each RNN layer as initial hidden
        h0 = torch.stack([h0_single_layer] * self.num_layers, dim=0)  # [L, B, H]

        # Project control inputs per time step
        x = self.input_proj(u.view(B * T, -1)).view(B, T, -1)

        # Run GRU
        rnn_out, _ = self.rnn(x, h0)

        # Output deltas and accumulate to absolute positions
        rnn_out = self.norm(rnn_out)
        deltas = self.delta_head(rnn_out)  # [B, T, 2]

        # Cumulative sum of deltas starting from p_ped0
        p_seq = []
        prev = p_ped0
        for t in range(T):
            prev = prev + deltas[:, t, :]
            p_seq.append(prev)
        p_ped_seq = torch.stack(p_seq, dim=1)

        return p_ped_seq


def compute_sequence_loss(
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute MSE loss over the full sequence and per-step average.

    Args:
        pred_positions: [B, T, 2]
        target_positions: [B, T, 2]

    Returns:
        total_mse: scalar loss over all elements
        per_step_mse: [T] mean MSE per step (detached)
    """
    assert pred_positions.shape == target_positions.shape
    B, T, D = pred_positions.shape

    mse_all = (pred_positions - target_positions) ** 2  # [B, T, 2]
    total_mse = mse_all.mean()

    with torch.no_grad():
        per_step_mse = mse_all.mean(dim=(0, 2))  # [T]

    return total_mse, per_step_mse


