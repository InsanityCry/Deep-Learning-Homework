from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        super().__init__()
        self.n_track = n_track
        self.n_waypoints = n_waypoints
        input_dim = n_track * 2 * 2  # left/right, each with 2 coords
        output_dim = n_waypoints * 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Concatenate left and right boundaries
        x = torch.cat([track_left, track_right], dim=1)  # (b, 2*n_track, 2)
        x = x.view(x.shape[0], -1)  # (b, 2*n_track*2)
        out = self.mlp(x)  # (b, n_waypoints*2)
        return out.view(-1, self.n_waypoints, 2)


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.query_embed = nn.Embedding(n_waypoints, d_model)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Concatenate left and right boundaries
        b = track_left.shape[0]
        boundaries = torch.cat([track_left, track_right], dim=1)  # (b, 2*n_track, 2)
        # Linear projection for input points
        if not hasattr(self, 'input_proj'):
            # Patch for legacy: add missing layers if not present
            self.d_model = 64
            self.input_proj = nn.Linear(2, self.d_model)
            decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=128, batch_first=True)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
            self.output_proj = nn.Linear(self.d_model, 2)
        memory = self.input_proj(boundaries)  # (b, 2*n_track, d_model)
        # Queries: (b, n_waypoints, d_model)
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        # Transformer decoder: queries attend to memory
        tgt = torch.zeros_like(queries)  # (b, n_waypoints, d_model)
        decoded = self.decoder(tgt=queries, memory=memory)  # (b, n_waypoints, d_model)
        out = self.output_proj(decoded)  # (b, n_waypoints, 2)
        return out


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()
        self.n_waypoints = n_waypoints
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)
        # CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, n_waypoints * 2),
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        x = self.features(x)
        x = self.head(x)
        return x.view(-1, self.n_waypoints, 2)


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m
def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
