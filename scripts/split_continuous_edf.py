"""Split an explicitly configured continuous EDF into per-seizure segments.

The authorized private configuration supplies the participant and selected trials.
Crop at onset +/- 60 seconds, clip to available bounds, and retain all channels
at native sample rate. The EDF is read lazily; no filtering is performed here.
Use --dry-run to inspect a plan and --smoke-test for the configured smoke trial.
"""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path
CONTINUOUS_PATIENT = str(private_value("continuous_recording", "patient"))
SINGLE_TRIAL_MANIFEST = str(private_value("continuous_recording", "single_trial_manifest_filename"))

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import mne
import numpy as np

ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
FINAL_AUDIT = source_root("data") / "annotations" / "clinical_annotations.xlsx"
TRIALS = private_value("continuous_recording", "trials")
PRE_S, POST_S = 60.0, 60.0


def assert_writable(out_dir: Path) -> None:
    """Never write under a configured read-only input root."""
    assert_output_path(out_dir)
    out_dir = out_dir.resolve()
    if out_dir == AUX_ROOT or AUX_ROOT in out_dir.parents:
        raise SystemExit(f"Refusing to write under read-only source root: {out_dir}")


def load_onsets() -> pd.DataFrame:
    df = pd.read_excel(FINAL_AUDIT, f"{CONTINUOUS_PATIENT}_data")
    required = {"patient_id", "trial_id", "onset_sec_from_edf_start", "edf_path", "t0_time"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Configured participant audit sheet missing required columns: {', '.join(missing)}")
    continuous = df[(df.patient_id == CONTINUOUS_PATIENT) & (df.trial_id.isin(TRIALS))]
    return (
        continuous[["trial_id", "onset_sec_from_edf_start", "edf_path", "t0_time"]]
        .rename(
            columns={
                "trial_id": "trial",
                "onset_sec_from_edf_start": "onset_sec_from_raw_edf_start",
                "edf_path": "raw_edf_path",
                "t0_time": "t0_time_for_audit",
            }
        )
        .sort_values("trial")
        .reset_index(drop=True)
    )


def _parse_datetime(value: object) -> datetime:
    if isinstance(value, pd.Timestamp):
        dt = value.to_pydatetime()
    elif isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).strip())
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _format_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def normalize_segment_timebase(seg: mne.io.BaseRaw, segment_t0_time: datetime) -> mne.io.BaseRaw:
    data = seg._data  # noqa: SLF001 - avoid an extra full copy of a large segment.
    fixed = mne.io.RawArray(data, seg.info.copy(), first_samp=0, copy="info", verbose="ERROR")
    fixed.set_meas_date(segment_t0_time)
    if len(seg.annotations) == 0:
        fixed.set_annotations(mne.Annotations(onset=[], duration=[], description=[], orig_time=segment_t0_time))
    else:
        original_orig_time = seg.annotations.orig_time or seg.info.get("meas_date")
        if original_orig_time is None:
            raise SystemExit("Cannot safely remap non-empty annotations without an absolute orig_time")
        original_orig_time = _parse_datetime(original_orig_time)
        fixed.set_annotations(
            mne.Annotations(
                onset=[
                    (original_orig_time + timedelta(seconds=float(onset)) - segment_t0_time).total_seconds()
                    for onset in seg.annotations.onset
                ],
                duration=seg.annotations.duration.copy(),
                description=seg.annotations.description.copy(),
                orig_time=segment_t0_time,
            )
        )
    if fixed.first_samp != 0 or fixed.first_time != 0.0:
        raise SystemExit("continuous segment timebase normalization failed")
    np.testing.assert_allclose(seg._data, fixed._data, rtol=0.0, atol=0.0)  # noqa: SLF001
    return fixed


def save_waveform_png(fif_path: Path, onset_sec: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    seg = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    sf = seg.info["sfreq"]
    a = max(0, int((onset_sec - 5) * sf))
    b = min(seg.n_times, int((onset_sec + 5) * sf))
    n = min(8, seg.info["nchan"])
    data = seg.get_data(picks=range(n), start=a, stop=b)
    t = np.arange(a, b) / sf
    plt.figure(figsize=(12, 8))
    for i, ch in enumerate(data):
        plt.plot(t, ch / (np.nanstd(ch) + 1e-12) + i * 6, lw=0.4)
    plt.axvline(onset_sec, color="r", ls="--", label="onset")
    plt.yticks([])
    plt.xlabel("time (s)")
    plt.title(fif_path.stem + f"  (first {n} ch, +/-5 s around onset)")
    plt.legend(loc="upper right")
    png = fif_path.with_suffix(".png")
    plt.savefig(png, dpi=110, bbox_inches="tight")
    plt.close()
    print("plot:", png)


def main() -> None:
    ap = argparse.ArgumentParser(description="Split continuous long EDF into per-seizure 10 kHz segments")
    ap.add_argument("--dry-run", action="store_true", help="print crop plan, read header only, write nothing")
    ap.add_argument("--smoke-test", action="store_true", help="only configured smoke trial, write to checks")
    ap.add_argument("--plot", action="store_true", help="save a waveform PNG around each onset")
    ap.add_argument("--out", default=None, help="override output directory")
    args = ap.parse_args()

    onsets = load_onsets()
    if args.smoke_test:
        onsets = onsets[onsets.trial == int(private_value("continuous_recording", "smoke_trial"))].reset_index(drop=True)
        out_dir = Path(args.out) if args.out else ROOT / "checks" / "continuous_smoke"
    else:
        out_dir = Path(args.out) if args.out else ROOT / "data_derived" / "segments" / CONTINUOUS_PATIENT
    assert_writable(out_dir)

    edf_path = onsets["raw_edf_path"].iloc[0]
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    sf = float(raw.info["sfreq"])
    tmax_file = float(raw.times[-1])

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    recs = []
    for _, r in onsets.iterrows():
        tt = int(r["trial"])
        onset_abs = float(r["onset_sec_from_raw_edf_start"])
        cs = max(0.0, onset_abs - PRE_S)
        ce = min(tmax_file, onset_abs + POST_S)
        onset_rel = onset_abs - cs
        raw_t0_time = _parse_datetime(r["t0_time_for_audit"])
        segment_t0_time = raw_t0_time + pd.to_timedelta(cs, unit="s").to_pytimedelta()
        fif_path = out_dir / f"{CONTINUOUS_PATIENT}_trial{tt:02d}_raw.fif"
        rec = {
            "patient": CONTINUOUS_PATIENT,
            "trial": tt,
            "onset_sec": round(onset_rel, 3),
            "onset_sec_from_raw_edf_start": onset_abs,
            "crop_start_sec_from_raw_edf_start": round(cs, 3),
            "crop_end_sec_from_raw_edf_start": round(ce, 3),
            "segment_t0_time": _format_iso(segment_t0_time),
            "raw_edf_t0_time": _format_iso(raw_t0_time),
            "actual_pre_s": round(onset_abs - cs, 3),
            "actual_post_s": round(ce - onset_abs, 3),
            "clipped": bool((ce < onset_abs + POST_S) or (cs > onset_abs - PRE_S)),
            "sfreq_hz": sf,
            "source_edf": str(edf_path),
            "out_file": str(fif_path),
        }
        if not args.dry_run:
            seg = raw.copy().crop(tmin=cs, tmax=ce)
            seg.load_data(verbose="ERROR")
            seg = normalize_segment_timebase(seg, segment_t0_time)
            rec["n_channels"] = int(seg.info["nchan"])
            rec["n_samples"] = int(seg.n_times)
            seg.save(fif_path, overwrite=True, verbose="ERROR")
            if args.plot:
                save_waveform_png(fif_path, onset_rel)
        recs.append(rec)

    man = pd.DataFrame(recs)
    cols = [c for c in ["trial", "onset_sec", "onset_sec_from_raw_edf_start",
                        "crop_start_sec_from_raw_edf_start", "crop_end_sec_from_raw_edf_start",
                        "actual_pre_s", "actual_post_s", "clipped", "n_channels", "n_samples"]
            if c in man.columns]
    print(man[cols].to_string(index=False))
    mode = "dry-run" if args.dry_run else ("smoke" if args.smoke_test else "full")
    print(f"\nmode={mode}  out={out_dir}")
    if not args.dry_run:
        man_path = out_dir / f"{CONTINUOUS_PATIENT}_crop_manifest.csv"
        man.to_csv(man_path, index=False, encoding="utf-8-sig")
        print("manifest:", man_path)


if __name__ == "__main__":
    main()
