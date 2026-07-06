"""
compute_parameter_count.py
===========================
Compute and report the number of parameters for EEGNet and
lightweightEEGNet (models.py), matching the model_kwargs used in train.py.

No trained weights are needed for this: parameter counts are a property
of the architecture (n_channels, n_samples, n_classes), not of the
learned values. Both models are freshly instantiated here.

Note on n_samples
------------------
- EEGNet uses AdaptiveAvgPool2d in its classifier, so its parameter
  count does NOT depend on n_samples (trial length).
- lightweightEEGNet infers its Linear layer's input size from a dummy
  forward pass, so its parameter count DOES depend on n_samples. If your
  three periods (BSL / SENS / DELAY) use different trial lengths, edit
  PERIOD_N_SAMPLES below with the actual values used in training
  (as read from the CSVs, per utils.run_cv -> n_samples=X.shape[-1]).

Usage
-----
    python compute_parameter_count.py
"""

from models import EEGNet, lightweightEEGNet


# =========================================================
# CONFIG — match train.py MODEL_KWARGS
# =========================================================

N_CHANNELS = 64
N_CLASSES  = 3

# Edit these to the real per-period trial lengths if they differ.
# 250 is the default used in models.py; adjust as needed.
PERIOD_N_SAMPLES = {
    "BSL":   250,
    "SENS":  250,
    "DELAY": 250,
}

EEGNET_KWARGS = {
    "n_channels":   N_CHANNELS,
    "n_classes":    N_CLASSES,
    "dropout_rate": 0.5,
}

LIGHTWEIGHT_KWARGS = {
    "n_channels": N_CHANNELS,
    "n_classes":  N_CLASSES,
}


# =========================================================
# HELPERS
# =========================================================

def count_parameters(model):
    """Return (total_params, trainable_params) for an nn.Module."""

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable


def print_breakdown(model, name):
    """Print a per-submodule parameter breakdown."""

    print(f"\n  Per-block breakdown ({name}):")

    for module_name, module in model.named_children():

        n = sum(p.numel() for p in module.parameters())

        if n > 0:
            print(f"    {module_name:<18s}: {n:>10,}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("=" * 55)
    print("PARAMETER COUNT SUMMARY")
    print("=" * 55)

    # ---- EEGNet (n_samples-independent) --------------------
    eegnet = EEGNet(n_samples=250, **EEGNET_KWARGS)
    total, trainable = count_parameters(eegnet)

    print(f"\nEEGNet  (n_channels={N_CHANNELS}, n_classes={N_CLASSES})")
    print(f"  Total parameters     : {total:,}")
    print(f"  Trainable parameters : {trainable:,}")
    print_breakdown(eegnet, "EEGNet")

    # ---- lightweightEEGNet (n_samples-dependent) ------------
    print(f"\nlightweightEEGNet  (n_channels={N_CHANNELS}, n_classes={N_CLASSES})")

    for period, n_samples in PERIOD_N_SAMPLES.items():

        model = lightweightEEGNet(n_samples=n_samples, **LIGHTWEIGHT_KWARGS)
        total, trainable = count_parameters(model)

        print(f"\n  [{period}] n_samples={n_samples}")
        print(f"    Total parameters     : {total:,}")
        print(f"    Trainable parameters : {trainable:,}")
        print_breakdown(model, f"lightweightEEGNet-{period}")

    print("\n" + "=" * 55)