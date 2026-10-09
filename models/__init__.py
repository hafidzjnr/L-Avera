# Registri model arsitektur ChickVoice-EFEST
from models.baseline_cnn import build_model as _baseline
from models.compact_cnn import build_model as _compact
from models.ds_cnn import build_model as _dscnn
from models.tcresnet import build_model as _tcresnet

# Pemetaan nama model ke fungsi pembangun arsitektur
REGISTRY = {
    "baseline_cnn": _baseline,
    "compact_cnn": _compact,
    "ds_cnn": _dscnn,
    "tc_resnet8": _tcresnet,
}

__all__ = ["REGISTRY"]
