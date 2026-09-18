from pathlib import Path


def resolve_backbone_weights(weights_path=None, *, config_directory=None):
    if not isinstance(weights_path, (str, Path)):
        raise ValueError('Set model.backbone_weights to a local model.safetensors file.')
    if isinstance(weights_path, str):
        weights_path = weights_path.strip()
    if not weights_path:
        raise ValueError('Set model.backbone_weights to a local model.safetensors file.')
    path = Path(weights_path).expanduser()
    if not path.is_absolute():
        path = Path(config_directory or Path.cwd()) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Backbone weights not found: {path}')
    return path
