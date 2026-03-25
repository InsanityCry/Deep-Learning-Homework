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
        input_dim = n_track * 2 * 2
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, n_waypoints * 2)
        )

    def forward(self, track_left, track_right, **kwargs):
        x = torch.cat([track_left, track_right], dim=1).view(track_left.shape[0], -1)
        return self.model(x).view(-1, self.n_waypoints, 2)

class TransformerPlanner(nn.Module):
    def __init__(self, n_track: int = 10, n_waypoints: int = 3, d_model: int = 128):
        super().__init__()
        self.n_waypoints = n_waypoints
        self.input_proj = nn.Linear(2, d_model)

        # Positional embedding for the input lane boundaries
        self.pos_embed = nn.Parameter(torch.randn(1, n_track * 2, d_model) * 0.1)

        # Learned query embeddings for the waypoints (latent array)
        self.query_embed = nn.Parameter(torch.randn(1, n_waypoints, d_model) * 0.1)

        # Perceiver-style cross-attention using TransformerDecoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dim_feedforward=512, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        self.output_proj = nn.Linear(d_model, 2)

    def forward(self, track_left, track_right, **kwargs):
        b = track_left.shape[0]

        # Keys and values: lane boundary features (byte array)
        combined = torch.cat([track_left, track_right], dim=1)
        memory = self.input_proj(combined) + self.pos_embed

        # Queries: target waypoint embeddings
        queries = self.query_embed.expand(b, -1, -1)

        # Cross-attention over the lane boundaries
        out = self.decoder(tgt=queries, memory=memory)

        return self.output_proj(out)

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
