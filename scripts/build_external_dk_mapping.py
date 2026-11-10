"""Prepare ds004100 DK template anatomy, then separate treatment counts.

Stages are explicit so anatomical QC can finish before the treatment join.
"""

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from seeg.external_dk import TemplateAtlas, compare_atlases, count_regions, map_pairs  # noqa: E402


ROOT = source_root("external_preparation")
SOURCE = (source_root("brainstorm") / "anat" / "@default_subject")
OUT = output_root() / "external" / "dk_mapping"
COLORS = {"gray": "#29a6ff", "white": "#ffcc55", "ventricle": "#53e5cb",
          "csf": "#53e5cb", "unknown": "#ff5bca", "non_neural": "#cd8fff",
          "mixed_neural_review": "#ff7147"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_template():
    mri = loadmat(SOURCE / "subjectimage_T1.mat", simplify_cells=True)
    dk = loadmat(SOURCE / "subjectimage_Deskian-Killiany_volatlas.mat", simplify_cells=True)
    aseg = loadmat(SOURCE / "subjectimage_ASEG_volatlas.mat", simplify_cells=True)
    original = nib.load(OUT / "provenance/mni_icbm152_t1_tal_nlin_asym_09c.nii")
    return mri, dk, aseg, original


def verify_geometry(mri, dk, aseg, original):
    """Establish voxel-grid identity using image contents, not filenames alone."""
    a = np.asarray(original.dataobj, dtype=float)
    b = np.asarray(mri["Cube"], dtype=float)
    if any(x["Cube"].shape != a.shape for x in (mri, dk, aseg)):
        raise ValueError("Original NIfTI and Brainstorm volume dimensions differ")
    if nib.aff2axcodes(original.affine) != ("R", "A", "S"):
        raise ValueError("Original volume is not in the expected RAS voxel orientation")
    if not np.allclose(original.header.get_zooms()[:3], mri["Voxsize"]):
        raise ValueError("Original and Brainstorm voxel sizes differ")
    use = (a > 0) | (b > 0)
    x, y = a[use], b[use]
    slope, intercept = np.polyfit(x, y, 1)
    corr = float(np.corrcoef(x, y)[0, 1])
    rmse = float(np.sqrt(np.mean((slope * x + intercept - y) ** 2)))
    # At most one intensity level of quantization discrepancy, no spatial fit.
    if corr < 0.9999 or rmse > 1:
        raise ValueError(f"Voxelwise template identity not established: r={corr}, RMSE={rmse}")
    for volume in (dk, aseg):
        if not np.array_equal(volume["Voxsize"], mri["Voxsize"]):
            raise ValueError("Atlas and T1 voxel sizes differ")
        for field in ("R", "T"):
            if not np.allclose(volume["SCS"][field], mri["SCS"][field]):
                raise ValueError("Atlas and T1 orientation metadata differ")
    return {
        "status": "VOXEL_GRID_IDENTITY_VERIFIED",
        "template": "Brainstorm ICBM152 2023b / ICBM152 nonlinear asymmetric 2009c",
        "original_nifti_url": "https://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/"
        "mni_icbm152_nlin_asym_09c_nifti.zip",
        "shape": list(a.shape), "voxel_mm": [float(v) for v in original.header.get_zooms()[:3]],
        "orientation": list(nib.aff2axcodes(original.affine)),
        "voxel_to_mni_mm_zero_based": original.affine.tolist(),
        "t1_voxelwise_correlation": corr, "intensity_fit_rmse_uint8": rmse,
        "intensity_fit_slope_intercept": [float(slope), float(intercept)],
        "spatial_resampling_or_registration_performed": False,
        "brainstorm_ncs_used": False,
        "brainstorm_ncs_reason": "NCS is an additional SPM/IXI affine; original ICBM world "
        "coordinates use the verified original NIfTI affine directly.",
        "input_coordinate_policy": "ICBM152 MNI coordinates in millimetres. "
        "README conflicts with BIDS fsaverage/metres; conflict retained.",
        "input_exact_icbm152_variant": "NOT_REPORTED_BY_DATASET",
        "individual_anatomy_validation": "UNAVAILABLE_NO_INDIVIDUAL_MRI",
        "precision_limit": "Template-based anatomical correspondence, not verified individual "
        "cortical folding or gray/white membership. Six axial coordinate perturbations "
        "at 1 and 2 mm are diagnostics, not confidence probabilities or validated error bounds.",
    }


def axial_qc(subject, rows, mri, atlas, path):
    valid = [r for r in rows if r["voxel_continuous"] is not None]
    xyz = np.asarray([r["midpoint_xyz"] for r in valid])
    bins = np.floor(xyz[:, 2] / 8 + 0.5).astype(int) * 8
    centers = np.arange(bins.min(), bins.max() + 1, 8)
    ncols, nrows = 5, int(np.ceil(len(centers) / 5))
    fig, axs = plt.subplots(nrows, ncols, figsize=(16, 3.0 * nrows), facecolor="#10141b")
    extent = [atlas.affine[0, 3] - 0.5,
              atlas.affine[0, 3] + atlas.cube.shape[0] - 0.5,
              atlas.affine[1, 3] - 0.5,
              atlas.affine[1, 3] + atlas.cube.shape[1] - 0.5]
    for ax, center in zip(np.ravel(axs), centers, strict=False):
        z = int(np.floor(atlas.to_voxel([0, 0, center])[2] + 0.5))
        ax.imshow(mri["Cube"][:, :, z].T, origin="lower", cmap="gray",
                  vmin=0, vmax=220, extent=extent)
        plane = atlas.cube[:, :, z].T
        # Thin label edges on the template, independent of treatment labels.
        edge = np.zeros(plane.shape, bool)
        edge[1:] |= (plane[1:] != plane[:-1]) & (plane[1:] > 0)
        edge[:, 1:] |= (plane[:, 1:] != plane[:, :-1]) & (plane[:, 1:] > 0)
        overlay = np.zeros((*edge.shape, 4))
        overlay[edge] = [0.2, 0.85, 0.8, 0.15]
        ax.imshow(overlay, origin="lower", extent=extent)
        selected = np.flatnonzero(bins == center)
        for i in selected:
            r = valid[i]
            # Segments link only each real adjacent pair, never missing contacts.
            e = np.asarray([r["endpoint1_xyz"], r["endpoint2_xyz"]], float)
            ax.plot(e[:, 0], e[:, 1], color="#eeeeee", alpha=0.3, lw=0.6)
        if len(selected):
            ax.scatter(xyz[selected, 0], xyz[selected, 1],
                       c=[COLORS[valid[i]["tissue_class"]] for i in selected],
                       s=17, linewidths=0.35, edgecolors="black", zorder=5)
        ax.set_title(f"z = {center:+d} mm | {len(selected)} pairs", color="white", fontsize=10)
        ax.set_xlim(-105, 105)
        ax.set_ylim(-115, 90)
        ax.set_xticks([-80, 0, 80])
        ax.set_yticks([-80, 0, 60])
        ax.tick_params(colors="#b6c3d0", labelsize=7)
        ax.set_xlabel("L              MNI x (mm)              R", color="#b6c3d0", fontsize=8)
    for ax in np.ravel(axs)[len(centers):]:
        ax.axis("off")
    counts = Counter(r["tissue_class"] for r in rows)
    text = " | ".join(f"{k}: {v}" for k, v in counts.items())
    fig.suptitle(f"{subject} | {len(rows)} common bipolar pairs\n{text}",
                 color="white", fontsize=15, y=0.995)
    legend = [Line2D([], [], marker="o", ls="", color=c, label=k)
              for k, c in COLORS.items() if k in counts]
    fig.legend(handles=legend, loc="lower center", ncol=6, facecolor="#e3e7eb", fontsize=9,
               bbox_to_anchor=(0.5, 0.020))
    fig.text(0.5, 0.003,
             "Template QC only. Each pair appears in one 8-mm slab; background is its central "
             "slice. Colors use exact midpoint voxels. No treatment labels.",
             ha="center", color="white", fontsize=9)
    fig.tight_layout(rect=[0, 0.055, 1, 0.94])
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return len(valid)


def anatomy_stage():
    out = OUT
    (out / "qc_images").mkdir(parents=True, exist_ok=True)
    mri, dk, aseg, original = load_template()
    contract = verify_geometry(mri, dk, aseg, original)
    labels = {int(v): str(name) for v, name, _ in dk["Labels"]}
    atlas = TemplateAtlas(dk["Cube"], original.affine, labels)
    pairs = read_json(ROOT / "stage2_common_pairs.json")
    if len(pairs) != 2637 or len({r["subject"] for r in pairs}) != 23:
        raise ValueError("Cohort changed: expected 23 subjects and 2637 bipolar pairs")
    rows = map_pairs(pairs, atlas)
    # Distance to any labelled neural volume is a QC flag, never a relocation.
    neural_ids = [v for v in labels if v not in {0, 24, 30, 31, 62, 63, 4, 5, 14, 15, 43, 44}]
    distances = distance_transform_edt(~np.isin(atlas.cube, neural_ids), sampling=mri["Voxsize"])
    for row in rows:
        ijk = row["voxel_index"]
        row["distance_to_labelled_neural_voxel_center_mm"] = (
            float(distances[tuple(ijk)]) if ijk is not None else None)
        row["aseg_value"] = int(aseg["Cube"][tuple(ijk)]) if ijk is not None else None
    sources = [ROOT / "stage2_common_pairs.json", ROOT / "coordinate_contracts.json",
               ROOT / "preprocessed_signal_manifest.json",
               SOURCE / "subjectimage_T1.mat",
               SOURCE / "subjectimage_Deskian-Killiany_volatlas.mat",
               SOURCE / "subjectimage_ASEG_volatlas.mat",
               out / "provenance/mni_icbm152_t1_tal_nlin_asym_09c.nii"]
    contract["source_sha256"] = {str(p): digest(p) for p in sources}
    contract["created_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(out / "geometry_contract.json", contract)
    save_json(out / "atlas_lookup.json", [{"id": i, "name": n} for i, n in labels.items()])
    save_json(out / "bipolar_anatomy_manifest.json", rows)
    # Copy the chosen atlas files so the mapping is reproducible after source updates.
    (out / "atlas_snapshot").mkdir(exist_ok=True)
    for name in ["subjectimage_T1.mat", "subjectimage_Deskian-Killiany_volatlas.mat",
                 "subjectimage_ASEG_volatlas.mat"]:
        shutil.copy2(SOURCE / name, out / "atlas_snapshot" / name)
    image_manifest = []
    for subject in sorted({r["subject"] for r in rows}):
        patient = [r for r in rows if r["subject"] == subject]
        path = out / "qc_images" / f"{subject}_axial.png"
        n = axial_qc(subject, patient, mri, atlas, path)
        image_manifest.append({"subject": subject, "path": str(path),
                               "n_pairs": len(patient), "n_pairs_shown": n,
                               "visual_review_status": "PENDING_ACTUAL_INSPECTION"})
    save_json(out / "qc_image_manifest.json", image_manifest)
    # Verify existing files only; no signal rewrite or re-filtering.
    signals = read_json(ROOT / "preprocessed_signal_manifest.json")
    if len(signals) != 86:
        raise ValueError("Expected 86 existing preprocessed signals")
    by_subject = {s: [r["bipolar_pair"] for r in rows if r["subject"] == s]
                  for s in {r["subject"] for r in rows}}
    for signal in signals:
        if digest(signal["signal_path"]) != signal["output_sha256"]:
            raise ValueError(f"Signal changed: {signal['signal_path']}")
        if signal["bipolar_channel_names"] != by_subject[signal["subject"]]:
            raise ValueError("Signal and anatomical channel orders differ")
    save_json(out / "stage_anatomy_summary.json", {
        "status": "ANATOMY_COMPUTED_VISUAL_QC_PENDING", "patients": 23,
        "common_pairs": len(rows), "existing_signal_hashes_verified": len(signals),
        "tissue_counts": dict(Counter(r["tissue_class"] for r in rows)),
        "mapping_reasons": dict(Counter(r["mapping_reason"] for r in rows)),
        "region_rows": len({(r["subject"], r["atlas_region"]) for r in rows
                            if r["region_model_eligible_tissue"]}),
        "anatomy_sha256": digest(out / "bipolar_anatomy_manifest.json"),
        "treatment_labels_read_in_this_stage": False,
        "suspected_region_rule": None,
    })
    print(json.dumps(read_json(out / "stage_anatomy_summary.json"), indent=2))


def counts_stage():
    qc = read_json(OUT / "visual_qc_review.json")
    if len(qc["patients"]) != 23 or any(not r["inspected"] for r in qc["patients"]):
        raise ValueError("Actual visual inspection of all 23 patients must precede counts")
    anatomy = read_json(OUT / "bipolar_anatomy_manifest.json")
    frozen = read_json(OUT / "stage_anatomy_summary.json")
    if frozen["anatomy_sha256"] != digest(OUT / "bipolar_anatomy_manifest.json"):
        raise ValueError("Anatomy changed after its label-blind freeze")
    pairs, regions = count_regions(anatomy, read_json(ROOT / "treatment_contact_labels.json"),
                                   read_json(ROOT / "bipolar_treatment_labels.json"))
    compared, transitions = compare_atlases(anatomy, read_json(ROOT / "bipolar_anatomy_manifest.json"))
    save_json(OUT / "bipolar_treatment_count_audit.json", pairs)
    save_json(OUT / "region_counts_all_tissues.json", regions)
    gray = [r for r in regions if r["region_model_eligible_tissue"]]
    save_json(OUT / "region_treatment_count_ranking_pending_rule.json", {
        "status": "AWAITING_USER_SUSPECTED_REGION_RULE", "rows": gray,
        "count_definition": "N=all common good pairs in this midpoint-defined region; "
        "B=both endpoints treated; M=exactly one endpoint treated. B+M+neither=N.",
        "ranking": "B descending; equal B tied; alphabetical display only within ties",
        "suspected_region_rule": None,
    })
    save_json(OUT / "aal_to_dk_pair_comparison.json", compared)
    save_json(OUT / "aal_to_dk_tissue_transitions.json", transitions)
    patients = []
    old_regions = read_json(ROOT / "region_treatment_count_ranking_pending_rule.json")["rows"]
    for subject in sorted({r["subject"] for r in anatomy}):
        a = [r for r in anatomy if r["subject"] == subject]
        g = [r for r in gray if r["subject"] == subject]
        o = [r for r in old_regions if r["subject"] == subject]
        patients.append({
            "subject": subject, "common_pairs": len(a),
            "dk_regions": len(g), "aal_regions": len(o),
            "dk_gray_pairs": sum(r["n_bipolar_pairs"] for r in g),
            "aal_gray_pairs": sum(r["n_bipolar_pairs"] for r in o),
            "dk_gray_both_treated": sum(r["n_both_end_treated_pairs"] for r in g),
            "aal_gray_both_treated": sum(r["n_both_end_treated_pairs"] for r in o),
            "dk_gray_mixed": sum(r["n_mixed_treatment_pairs"] for r in g),
            "tissue_counts": dict(Counter(r["tissue_class"] for r in a)),
            "dk_gray_regions_with_B_gt_0": sum(r["n_both_end_treated_pairs"] > 0 for r in g),
            "anatomical_boundary_flags_1mm": sum(
                r["axis_1mm_same_label_fraction"] is not None
                and r["axis_1mm_same_label_fraction"] < 1 for r in a),
        })
    save_json(OUT / "patient_summary.json", patients)
    save_json(OUT / "counts_summary.json", {
        "status": "DK_TEMPLATE_MAPPING_AND_COUNTS_COMPLETE_RULE_PENDING",
        "patients": len(patients), "common_pairs": len(pairs), "gray_regions": len(gray),
        "gray_pairs": sum(r["n_bipolar_pairs"] for r in gray),
        "gray_both_treated_pairs": sum(r["n_both_end_treated_pairs"] for r in gray),
        "gray_mixed_pairs": sum(r["n_mixed_treatment_pairs"] for r in gray),
        "source_sha256": {str(ROOT / n): digest(ROOT / n) for n in
                          ["treatment_contact_labels.json", "bipolar_treatment_labels.json",
                           "label_source_sha256.json"]},
        "visual_review_sha256": digest(OUT / "visual_qc_review.json"),
        "anatomy_sha256": digest(OUT / "bipolar_anatomy_manifest.json"),
        "treatment_counts_are_not_epileptogenic_ground_truth": True,
        "labels_assigned": False, "models_run": False, "network_features_computed": False,
    })
    print(json.dumps(read_json(OUT / "counts_summary.json"), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["anatomy", "counts"])
    args = parser.parse_args()
    if args.stage == "anatomy":
        anatomy_stage()
    else:
        counts_stage()
