import numpy as np
import torch
import torch.nn as nn

CLASS_NAMES = [
    "Water", "Trees", "Grass", "Flooded Vegetation", "Crops",
    "Scrub/Shrub", "Built Area", "Bare Ground", "Snow/Ice"
]

NUM_CLASSES = 9

_model = None
_device = None


class PrithviClassifier(nn.Module):
    """Prithvi backbone + linear classifier head for land cover."""

    def __init__(self, backbone, num_classes=9):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, list):
            features = features[-1]
        if features.dim() == 3:
            features = features.mean(dim=1)
        elif features.dim() == 4:
            features = features.mean(dim=[2, 3])
        return self.head(features)


def get_model():
    global _model, _device
    if _model is not None:
        return _model, _device

    try:
        from terratorch.registry import BACKBONE_REGISTRY
        import terratorch

        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        backbone = BACKBONE_REGISTRY.build(
            "prithvi_eo_v2_100_tl",
            pretrained=True,
            num_frames=1
        )

        _model = PrithviClassifier(backbone, num_classes=NUM_CLASSES)
        _model = _model.to(_device)
        _model.eval()

        return _model, _device

    except Exception as e:
        raise RuntimeError(f"Failed to load Prithvi model: {e}")


HLS_BAND_MEAN = np.array([0.0390, 0.0542, 0.0695, 0.2265, 0.2457, 0.1786])
HLS_BAND_STD = np.array([0.0275, 0.0349, 0.0546, 0.1264, 0.1273, 0.1099])

PRITHVI_BAND_INDICES = {
    'BLUE': 0,
    'GREEN': 1,
    'RED': 2,
    'NIR_NARROW': 3,
    'SWIR_1': 4,
    'SWIR_2': 5,
}


def select_prithvi_bands(s2_array: np.ndarray) -> np.ndarray:
    """Extract 6 Prithvi bands from 9-band Sentinel-2 array.

    S2 bands: B2(Blue), B3(Green), B4(Red), B5(RedEdge1), B6(RedEdge2),
              B7(RedEdge3), B8(NIR), B11(SWIR1), B12(SWIR2)

    Prithvi HLS bands: Blue, Green, Red, NIR_NARROW, SWIR_1, SWIR_2
    """
    h, w = s2_array.shape[:2]

    if s2_array.shape[2] >= 9:
        blue = s2_array[:, :, 0]
        green = s2_array[:, :, 1]
        red = s2_array[:, :, 2]
        nir = s2_array[:, :, 6]
        swir1 = s2_array[:, :, 7]
        swir2 = s2_array[:, :, 8]
    else:
        raise ValueError(f"Expected 9 bands, got {s2_array.shape[2]}")

    bands = np.stack([blue, green, red, nir, swir1, swir2], axis=-1)

    bands = bands.astype(np.float32)
    if bands.max() > 10000:
        bands = bands / 10000.0

    bands = (bands - HLS_BAND_MEAN) / (HLS_BAND_STD + 1e-8)

    return bands


def run_prithvi(s2_array: np.ndarray) -> np.ndarray:
    """Run Prithvi inference on Sentinel-2 array.

    Args:
        s2_array: (H, W, 9) Sentinel-2 L1C array

    Returns:
        probs: (H, W, 9) softmax probabilities
    """
    model, device = get_model()

    bands = select_prithvi_bands(s2_array)
    h, w, c = bands.shape

    patch_size = 224
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    if pad_h > 0 or pad_w > 0:
        bands_padded = np.pad(bands, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        bands_padded = bands

    hp, wp, _ = bands_padded.shape

    all_probs = np.zeros((hp, wp, NUM_CLASSES), dtype=np.float32)
    count = np.zeros((hp, wp), dtype=np.float32)

    for y in range(0, hp, patch_size):
        for x in range(0, wp, patch_size):
            patch = bands_padded[y:y+patch_size, x:x+patch_size]
            patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            patch_tensor = patch_tensor.unsqueeze(2)

            with torch.no_grad():
                logits = model(patch_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                probs_2d = probs.reshape(NUM_CLASSES, 1, 1)
                all_probs[y:y+patch_size, x:x+patch_size] += probs_2d.transpose(1, 2, 0)
                count[y:y+patch_size, x:x+patch_size] += 1

    count = np.maximum(count, 1)
    all_probs = all_probs / count[:, :, np.newaxis]

    return all_probs[:h, :w]


def classify_with_prithvi(s2_array: np.ndarray, confidence_threshold: float = 0.0) -> dict:
    """Run Prithvi classification and return results in same format as DW."""
    probs = run_prithvi(s2_array)

    class_map = np.argmax(probs, axis=-1)
    confidence_map = np.max(probs, axis=-1)

    if confidence_threshold > 0:
        low_conf = confidence_map < confidence_threshold
        class_map[low_conf] = -1

    return {
        "probs": probs,
        "class_map": class_map,
        "confidence_map": confidence_map
    }
