#!/usr/bin/env python3
"""
Quick CLI to fit custom resonator data with scresonators.

Supports:
- Frequency array from .npy or CSV
- S21 complex array from .npy (complex dtype) or CSV with re,im or mag,phase(deg)
- Methods: DCM (hanger/coupled transmission) or Reflection

Usage examples:

  python examples/fit_custom.py \
    --freq examples/resonator-spectroscopy.input.frequencies.npy \
    --s21  examples/calculate-signal-magnitude-and-phase.output.data-raw.npy \
    --method dcm --plot

If your frequencies are in Hz, add: --freq-units Hz
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Ensure repo root is on sys.path if running from examples/
try:
    HERE = Path(__file__).resolve().parent
    for cand in [HERE, *HERE.parents]:
        if (cand / 'src').exists():
            sys.path.insert(0, str(cand))
            break
except Exception:
    pass

# Try local (repo) imports first
try:
    from src.resonator import Resonator
    from src.fit_methods.dcm import DCM
    from src.fit_methods.reflection import ReflectionMode
except Exception:
    # Fallback to installed package layout if available
    try:
        from scresonators.resonator import Resonator  # type: ignore
        from scresonators.fit_methods.dcm import DCM  # type: ignore
        from scresonators.fit_methods.reflection import ReflectionMode  # type: ignore
    except Exception:
        try:
        from fit_resonator.resonator import Resonator  # type: ignore
        from fit_resonator.fit_methods.dcm import DCM  # type: ignore
        from fit_resonator.fit_methods.reflection import ReflectionMode  # type: ignore
        except Exception as e:  # pragma: no cover
            print("Could not import scresonators modules. Run from repo root or install the package.")
            raise


def load_array(path: Path) -> np.ndarray:
    """Load array from .npy or .csv/.txt"""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    # CSV or TXT
    return np.genfromtxt(path, delimiter=",", dtype=float)


def parse_s21(data: np.ndarray, fmt: str) -> np.ndarray:
    """Normalize S21 inputs to complex ndarray of shape (N,)"""
    if fmt == "complex_npy":
        if not np.iscomplexobj(data):
            raise ValueError("Expected complex dtype in npy for complex_npy format.")
        return data.astype(np.complex128)
    if fmt == "reim_csv":
        if data.ndim == 1:
            raise ValueError("reim_csv expects 2 columns: re,im")
        return (data[:, 0] + 1j * data[:, 1]).astype(np.complex128)
    if fmt == "magphase_deg_csv":
        if data.ndim == 1:
            raise ValueError("magphase_deg_csv expects 2 columns: mag,phase_deg")
        mag = data[:, 0]
        ph = np.deg2rad(data[:, 1])
        return (mag * np.exp(1j * ph)).astype(np.complex128)
    raise ValueError(f"Unknown s21-format: {fmt}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Fit resonator data with scresonators")
    p.add_argument("--freq", required=False, type=Path, help="Path to frequencies .npy or .csv")
    p.add_argument("--s21", required=False, type=Path, help="Path to S21 .npy or .csv")
    p.add_argument("--touchstone", type=Path, default=None, help="Alternative: .s2p/.s1p Touchstone file")
    p.add_argument("--s21-format", default="complex_npy", choices=["complex_npy", "reim_csv", "magphase_deg_csv"],
                   help="Format of S21 file if not complex npy")
    p.add_argument("--method", default="dcm", choices=["dcm", "reflection"], help="Fit method")
    p.add_argument("--freq-units", default="GHz", choices=["GHz", "Hz", "MHz", "kHz"], help="Units of input freqs")
    p.add_argument("--no-delay-removal", action="store_true", help="Disable electrical delay removal pre-process")
    p.add_argument("--plot", action="store_true", help="Show a quick plot of data and fit")
    args = p.parse_args(argv)

    if args.touchstone is None and (args.freq is None or args.s21 is None):
        p.error("Either --touchstone or both --freq and --s21 must be provided")

    # Build resonator
    res = Resonator()

    # Load data
    if args.touchstone is not None:
        res.load_data_from_touchstone(str(args.touchstone))
    else:
        f = load_array(args.freq).astype(float)
        sraw = load_array(args.s21)

        # Convert freq units to GHz
        unit_scale = {"Hz": 1e-9, "kHz": 1e-6, "MHz": 1e-3, "GHz": 1.0}[args.freq_units]
        f_ghz = f * unit_scale

        # If CSV with one column, ensure shape (N,)
        if f_ghz.ndim > 1:
            f_ghz = np.squeeze(f_ghz)
        if sraw.ndim > 1 and args.s21_format == "complex_npy":
            # user probably passed CSV; fall back to re,im format by default
            s = parse_s21(sraw, "reim_csv")
        else:
            s = parse_s21(sraw, args.s21_format)

        # Basic shape check
        if f_ghz.shape[0] != s.shape[0]:
            raise ValueError(f"freq and s21 length mismatch: {f_ghz.shape} vs {s.shape}")

        res.load_data(fdata=f_ghz, sdata=s)

    # Set method
    if args.method == "dcm":
        res.set_fitting_strategy(strategy=DCM())
    else:
        res.set_fitting_strategy(strategy=ReflectionMode())

    # Control preprocessing
    if args.no_delay_removal:
        res.fitter.remove_elec_delay = False

    # Fit
    params = res.fit(verbose=True)

    # Print results
    def pval(name):
        v = params[name].value
        se = params[name].stderr
        return v, se

    keys = [k for k in ["f0", "Q", "Qc", "Qi", "phi"] if k in params]
    print("\nFit parameters:")
    for k in keys:
        v, se = pval(k)
        se_str = f" ± {se:.3g}" if se is not None else ""
        print(f"  {k}: {v:.9g}{se_str}")

    # Optional plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            # Compute fitted curve on input grid
            method = res.fitter.fit_method
            if hasattr(method, "fit_function"):
                s_fit = method.fit_function(res.fdata, params)
            else:
                s_fit = method.func(res.fdata, **{k: params[k].value for k in params.keys() if k in ["Q", "Qc", "f0", "phi"]})

            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            # Magnitude vs freq
            ax[0].plot(res.fdata, 20*np.log10(np.abs(res.sdata)), label="data")
            ax[0].plot(res.fdata, 20*np.log10(np.abs(s_fit)), label="fit", ls="--")
            ax[0].set_xlabel("f (GHz)")
            ax[0].set_ylabel("|S21| (dB)")
            ax[0].legend()
            # Smith chart
            ax[1].plot(np.real(res.sdata), np.imag(res.sdata), label="data")
            ax[1].plot(np.real(s_fit), np.imag(s_fit), label="fit", ls="--")
            ax[1].set_aspect("equal", adjustable="box")
            ax[1].set_xlabel("Re")
            ax[1].set_ylabel("Im")
            ax[1].legend()
            fig.suptitle("scresonators fit")
            plt.tight_layout()
            plt.show()
        except Exception as e:  # pragma: no cover
            print(f"Plotting failed: {e}")


if __name__ == "__main__":
    sys.exit(main())
