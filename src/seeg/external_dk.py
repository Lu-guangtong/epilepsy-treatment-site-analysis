"""Label-blind template mapping and separate treatment-count audit for ds004100.

The atlas is a standard template, not an individual anatomical segmentation.
No nearest-gray expansion, label optimization, or supervised targets are used.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np


GRAY_DEEP = {8, 10, 11, 12, 13, 17, 18, 26, 47, 49, 50, 51, 52, 53, 54, 58}
WHITE = {2, 7, 41, 46, 77, 85, 251, 252, 253, 254, 255}
VENTRICLE = {4, 5, 14, 15, 43, 44}
NON_NEURAL = {30, 31, 62, 63}


def tissue_class(value):
    """Classify actual DK+aseg IDs; unmapped IDs never default to gray."""
    if 1001 <= value <= 1035 or 2001 <= value <= 2035:
        if value not in {1004, 2004}:  # corpus callosum in older lookup tables
            return "gray"
    if value in GRAY_DEEP:
        return "gray"
    if value in WHITE or value in {1004, 2004}:
        return "white"
    if value in VENTRICLE:
        return "ventricle"
    if value == 24:
        return "csf"
    if value in NON_NEURAL:
        return "non_neural"
    if value in {16, 28, 60}:
        return "mixed_neural_review"
    return "unknown"


def keyed(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[f] for f in fields)
        if key in result:
            raise ValueError(f"Duplicate key: {key}")
        result[key] = row
    return result


@dataclass
class TemplateAtlas:
    cube: np.ndarray
    affine: np.ndarray  # zero-based voxel centers -> original MNI world mm
    labels: dict

    def __post_init__(self):
        if self.cube.ndim != 3 or self.affine.shape != (4, 4):
            raise ValueError("Atlas must be 3D with a 4x4 voxel-to-world affine")
        if not np.isfinite(self.affine).all():
            raise ValueError("Nonfinite atlas affine")
        self.inverse = np.linalg.inv(self.affine)
        missing = set(np.unique(self.cube)) - set(self.labels)
        if missing:
            raise ValueError(f"Cube IDs absent from lookup table: {missing}")

    def to_voxel(self, xyz):
        xyz = np.asarray(xyz, dtype=float)
        return xyz @ self.inverse[:3, :3].T + self.inverse[:3, 3]

    def to_world(self, ijk):
        ijk = np.asarray(ijk, dtype=float)
        return ijk @ self.affine[:3, :3].T + self.affine[:3, 3]

    def lookup(self, xyz):
        empty = {
            "atlas_value": None, "atlas_region": "Unknown",
            "tissue_class": "unknown", "voxel_index": None,
            "voxel_continuous": None,
        }
        xyz = np.asarray(xyz, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            return dict(empty, mapping_reason="missing_coordinates")
        vox = self.to_voxel(xyz)
        # Nearest voxel center, with half ties toward the larger index.
        idx = np.floor(vox + 0.5).astype(int)
        if ((idx < 0) | (idx >= np.array(self.cube.shape))).any():
            return dict(empty, voxel_continuous=vox.tolist(),
                        mapping_reason="outside_template_volume")
        value = int(self.cube[tuple(idx)])
        return {
            "atlas_value": value, "atlas_region": self.labels[value],
            "tissue_class": tissue_class(value), "voxel_index": idx.tolist(),
            "voxel_continuous": vox.tolist(),
            "mapping_reason": "unlabelled_template_voxel" if value == 0
            else "nearest_voxel_center",
        }


def map_pairs(pairs, atlas):
    """Accept only geometry fields, making label columns unavailable by design."""
    fields = ["subject", "bipolar_pair", "shaft", "number_1", "contact_1",
              "contact_2", "endpoint1_xyz", "endpoint2_xyz", "midpoint_xyz",
              "coordinates_complete", "distance_mm", "included_runs",
              "network_eligible"]
    keyed(pairs, ("subject", "bipolar_pair"))
    result = []
    for pair in pairs:
        row = {f: pair[f] for f in fields}
        ep1 = atlas.lookup(pair["endpoint1_xyz"])
        ep2 = atlas.lookup(pair["endpoint2_xyz"])
        middle = atlas.lookup(pair["midpoint_xyz"])
        if middle["mapping_reason"] != "missing_coordinates":
            calculated = (np.asarray(pair["endpoint1_xyz"], float)
                          + np.asarray(pair["endpoint2_xyz"], float)) / 2
            if not np.allclose(calculated, pair["midpoint_xyz"], atol=1e-8):
                raise ValueError(f"Midpoint mismatch: {pair['bipolar_pair']}")
        row.update(middle)
        row.update({
            "atlas": "Brainstorm_ICBM152_2023b_Desikan_Killiany_aparc_aseg",
            "coordinate_space": "ICBM152_MNI_mm",
            "endpoint1_atlas": ep1, "endpoint2_atlas": ep2,
            "region_model_eligible_tissue": middle["tissue_class"] == "gray",
            "is_white_csf_ventricle_unknown": middle["tissue_class"] in
            {"white", "csf", "ventricle", "unknown"},
            "endpoint_midpoint_same_gray_region": (
                middle["tissue_class"] == "gray"
                and ep1["atlas_value"] == middle["atlas_value"]
                and ep2["atlas_value"] == middle["atlas_value"]),
            "localization_scope": "standard_template_not_individual_MRI",
        })
        for radius in (1, 2):
            if middle["mapping_reason"] == "missing_coordinates":
                row[f"axis_{radius}mm_same_label_fraction"] = None
                row[f"axis_{radius}mm_same_tissue_fraction"] = None
                continue
            perturbations = [atlas.lookup(np.asarray(pair["midpoint_xyz"])
                             + sign * radius * axis)
                             for axis in np.eye(3) for sign in (-1, 1)]
            row[f"axis_{radius}mm_same_label_fraction"] = sum(
                p["atlas_value"] == middle["atlas_value"] for p in perturbations) / 6
            row[f"axis_{radius}mm_same_tissue_fraction"] = sum(
                p["tissue_class"] == middle["tissue_class"] for p in perturbations) / 6
        result.append(row)
    return result


def count_regions(anatomy, contact_labels, old_pair_treatment):
    """Join frozen treatment after geometry; mixed pairs enter N but not B."""
    contacts = keyed(contact_labels, ("subject", "contact"))
    previous = keyed(old_pair_treatment, ("subject", "bipolar_pair"))
    if set(previous) != set(keyed(anatomy, ("subject", "bipolar_pair"))):
        raise ValueError("Anatomy/treatment pair keys do not match")
    pair_rows, grouped = [], {}
    for row in anatomy:
        key = row["subject"], row["bipolar_pair"]
        endpoint = [contacts[(row["subject"], row[c])]
                    for c in ("contact_1", "contact_2")]
        if any(not p["source_consistent"] or p["treated"] not in (0, 1)
               for p in endpoint):
            raise ValueError(f"Missing or inconsistent treatment source: {key}")
        a, b = (p["treated"] for p in endpoint)
        if (a, b) != (previous[key]["endpoint_1_treated"],
                      previous[key]["endpoint_2_treated"]):
            raise ValueError(f"Frozen treatment mismatch: {key}")
        both, mixed, neither = int(a == b == 1), int(a != b), int(a == b == 0)
        pair_rows.append({
            "subject": key[0], "bipolar_pair": key[1],
            "atlas_region": row["atlas_region"], "atlas_value": row["atlas_value"],
            "tissue_class": row["tissue_class"],
            "endpoint_1_treated": a, "endpoint_2_treated": b,
            "both_end_treated": both, "mixed_treatment": mixed,
            "neither_end_treated": neither, "either_end_treated": both + mixed,
            "region_denominator_contribution": 1,
            "suspected_region_label": None,
        })
        group_key = (row["subject"], row["atlas_region"], row["tissue_class"])
        if group_key not in grouped:
            grouped[group_key] = {
                "subject": key[0], "atlas_region": row["atlas_region"],
                "atlas_value": row["atlas_value"], "tissue_class": row["tissue_class"],
                "region_model_eligible_tissue": row["region_model_eligible_tissue"],
                "n_bipolar_pairs": 0, "n_both_end_treated_pairs": 0,
                "n_mixed_treatment_pairs": 0, "n_neither_end_treated_pairs": 0,
                "n_either_end_treated_pairs": 0, "bipolar_pairs": [],
                "suspected_region_label": None,
            }
        reg = grouped[group_key]
        reg["n_bipolar_pairs"] += 1
        reg["n_both_end_treated_pairs"] += both
        reg["n_mixed_treatment_pairs"] += mixed
        reg["n_neither_end_treated_pairs"] += neither
        reg["n_either_end_treated_pairs"] += both + mixed
        reg["bipolar_pairs"].append(key[1])
    all_regions = sorted(grouped.values(), key=lambda r: (
        r["subject"], not r["region_model_eligible_tissue"],
        -r["n_both_end_treated_pairs"], r["atlas_region"]))
    by_subject = defaultdict(list)
    for r in all_regions:
        assert r["n_bipolar_pairs"] == (r["n_both_end_treated_pairs"]
                                       + r["n_mixed_treatment_pairs"]
                                       + r["n_neither_end_treated_pairs"])
        if r["region_model_eligible_tissue"]:
            by_subject[r["subject"]].append(r)
    for regs in by_subject.values():
        counts = [r["n_both_end_treated_pairs"] for r in regs]
        for r in regs:
            r["burn_count_competition_rank"] = 1 + sum(
                v > r["n_both_end_treated_pairs"] for v in counts)
    return pair_rows, all_regions


def compare_atlases(anatomy, old_anatomy):
    old = keyed(old_anatomy, ("subject", "bipolar_pair"))
    new = keyed(anatomy, ("subject", "bipolar_pair"))
    if set(old) != set(new):
        raise ValueError("AAL/DK cohort or pair key mismatch")
    transitions = Counter()
    pair_rows = []
    for key, b in new.items():
        a = old[key]
        transitions[a["tissue_class"], b["tissue_class"]] += 1
        pair_rows.append({
            "subject": key[0], "bipolar_pair": key[1],
            "aal_region": a["atlas_region"], "dk_region": b["atlas_region"],
            "aal_tissue": a["tissue_class"], "dk_tissue": b["tissue_class"],
            "tissue_changed": a["tissue_class"] != b["tissue_class"],
            "gray_eligibility_changed": (a["tissue_class"] == "gray")
            != (b["tissue_class"] == "gray"),
        })
    summary = [{"aal_tissue": a, "dk_tissue": b, "n_pairs": n}
               for (a, b), n in sorted(transitions.items())]
    return pair_rows, summary
