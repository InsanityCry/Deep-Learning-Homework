# Segments of code may be written with the aid of AI tools
from pathlib import Path
import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]

class MLPPlanner(nn.Module):
    def __init__(self, n_track: int = 10, n_waypoints: int = 3):
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        # Use left/right boundaries, centerline, and lane width as inputs.
        input_dim = n_track * 2 * 4
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, n_waypoints * 2)
        )

    def _centerline_anchor(self, track_left: torch.Tensor, track_right: torch.Tensor) -> torch.Tensor:
        center = 0.5 * (track_left + track_right)

        idx = torch.linspace(1, self.n_track - 1, self.n_waypoints, device=center.device)
        idx0 = idx.floor().long().clamp(max=self.n_track - 1)
        idx1 = (idx0 + 1).clamp(max=self.n_track - 1)
        alpha = (idx - idx0.float()).view(1, -1, 1)

        p0 = center[:, idx0]
        p1 = center[:, idx1]
        return p0 * (1.0 - alpha) + p1 * alpha

    def forward(self, track_left, track_right, **kwargs):
        center = 0.5 * (track_left + track_right)
        width = track_right - track_left
        anchor = self._centerline_anchor(track_left, track_right)

        x = torch.cat([track_left, track_right, center, width], dim=2).reshape(track_left.shape[0], -1)
        residual = self.model(x).view(-1, self.n_waypoints, 2)
        return anchor + residual

class TransformerPlanner(nn.Module):
    def __init__(self, n_track: int = 10, n_waypoints: int = 3, d_model: int = 128):
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.input_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.side_embed = nn.Embedding(2, d_model)
        self.anchor_proj = nn.Linear(2, d_model)

        # Positional embedding for the input lane boundaries
        self.pos_embed = nn.Parameter(torch.randn(1, n_track * 2, d_model) * 0.1)

        # Learned query embeddings for the waypoints (latent array)
        self.query_embed = nn.Parameter(torch.randn(1, n_waypoints, d_model) * 0.1)

        # Perceiver-style cross-attention using TransformerDecoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
        )

    def _centerline_anchor(self, track_left: torch.Tensor, track_right: torch.Tensor) -> torch.Tensor:
        center = 0.5 * (track_left + track_right)

        idx = torch.linspace(1, self.n_track - 1, self.n_waypoints, device=center.device)
        idx0 = idx.floor().long().clamp(max=self.n_track - 1)
        idx1 = (idx0 + 1).clamp(max=self.n_track - 1)
        alpha = (idx - idx0.float()).view(1, -1, 1)

        p0 = center[:, idx0]
        p1 = center[:, idx1]
        return p0 * (1.0 - alpha) + p1 * alpha

    def forward(self, track_left, track_right, **kwargs):
        b = track_left.shape[0]

        # Keys and values: lane boundary features (byte array)
        combined = torch.cat([track_left, track_right], dim=1)
        memory = self.input_proj(combined)

        side_ids = torch.cat([
            torch.zeros(self.n_track, device=combined.device, dtype=torch.long),
            torch.ones(self.n_track, device=combined.device, dtype=torch.long),
        ]).view(1, -1)
        memory = memory + self.side_embed(side_ids) + self.pos_embed

        anchor = self._centerline_anchor(track_left, track_right)

        # Queries: learned embedding plus projected centerline anchor
        queries = self.query_embed.expand(b, -1, -1) + self.anchor_proj(anchor)

        # Cross-attention over the lane boundaries
        out = self.decoder(tgt=queries, memory=memory)

        residual = self.output_proj(out)
        return anchor + residual

class CNNPlanner(nn.Module):
    def __init__(self, n_waypoints: int = 3):
        super().__init__()
        self.n_waypoints = n_waypoints
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, n_waypoints * 2)
        )

    def forward(self, image, **kwargs):
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        f = self.features(x).view(x.shape[0], -1)
        return self.classifier(f).view(-1, self.n_waypoints, 2)

def save_model(model):
    from torch import save
    from os import path
    name = model.__class__.__name__.lower().replace('planner', '_planner')
    save(model.state_dict(), path.join(HOMEWORK_DIR, f'{name}.th'))

def load_model(model_name, device='cpu', with_weights=True):
    from torch import load
    from os import path
    if model_name == 'mlp_planner':
        model = MLPPlanner()
    elif model_name == 'transformer_planner':
        model = TransformerPlanner()
    elif model_name == 'cnn_planner':
        model = CNNPlanner()
    else:
        raise ValueError(f"Unknown model {model_name}")

    if with_weights:
        model_path = path.join(HOMEWORK_DIR, f'{model_name}.th')
        if path.exists(model_path):
            model.load_state_dict(load(model_path, map_location=device))
    return model.to(device)
