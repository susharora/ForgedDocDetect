#!/usr/bin/env python3
"""
build_manifest.py  (v3)  --  Phase 0B measurement spine for the FantasyID dissertation.

Walks a FantasyID tree (train/ and/or test/), pairs each image with its ground-truth
JSON, and emits two CSVs:

  manifest.csv   one row per IMAGE
  regions.csv    one row per REGION (original + altered)

v3 changes (all ADDITIVE — nothing that 0C's splits.csv depends on is altered:
 split / card_stem / language / attack_method_folder / manip_type / card_in_train
 are unchanged, so the frozen split and its sha256 remain valid):

  regions.csv gains, per box:
      x1_eff, y1_eff, x2_eff, y2_eff   intersection of the raw box with the image
      bbox_visible_fraction            clipped_area / raw_area  (1.0 = fully inside)
      bbox_truncated                   True when the raw box extends past the image edge
    The RAW x/y/width/height are preserved unchanged — clipping never overwrites source.
    No box is excluded; truncated boxes are clipped and their visible fraction recorded.

  manifest.csv gains, per image:
      device_raw                       verbatim device folder (huawei|iphone15|iphone15pro|scan)
      device                           canonical device (iphone15 -> iphone15pro)   [as v2]
      json_parse_ok                    renamed from has_valid_annotation (JSON parsed OK)
      any_bbox_truncated               any region in this image truncated by the crop
      altered_field_names_unique       de-duplicated altered field set (face|face -> face)
      capture_id                       card_stem::device_raw  (groups one physical capture)
      paired_bonafide_path             bonafide image with the same card_stem+device_raw
      paired_bonafide_available        whether such a bonafide exists

Usage:
  pip install pandas pillow
  python build_manifest.py --root "C:\\Users\\senor\\09DISS\\FANTASY" --out-dir ".\\manifest_out"
  # add --no-hash to skip sha256 (faster).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

try:
    from PIL import Image
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DEVICES = {"huawei", "iphone15pro", "iphone15", "scan"}
DEVICE_ALIAS = {"iphone15": "iphone15pro"}     # canonical grouping; raw label preserved separately
SPLITS = {"train", "test"}                     # anything else (e.g. examples/) is excluded
FACE_FIELDS = {"face"}
CROP_TOL_FRAC = 0.01
CROP_TOL_MIN = 3


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_size(path: Path):
    if not (_HAVE_PIL and path.exists()):
        return (None, None)
    try:
        with Image.open(path) as im:
            return im.size  # (w, h), header-only read
    except Exception:
        return (None, None)


def derive_path_fields(img_rel: Path):
    parts = [p.lower() for p in img_rel.parts]
    split = next((p for p in parts if p in SPLITS), "unknown")
    if "bonafide" in parts:
        class_label, class_folder = 0, "bonafide"
    elif "attack" in parts:
        class_label, class_folder = 1, "attack"
    else:
        class_label, class_folder = -1, "unknown"
    device_raw = next((p for p in parts if p in DEVICES), "unknown")
    attack_method = ""
    if class_folder == "attack":
        i = parts.index("attack")
        if i + 1 < len(parts) and parts[i + 1] not in DEVICES:
            attack_method = parts[i + 1]
    return split, class_label, class_folder, attack_method, device_raw


def split_stem(stem: str):
    if "-" in stem:
        lang, rest = stem.split("-", 1)
        return lang.lower(), rest
    return "unknown", stem


def classify_manip(altered_fields):
    has_face = any(f.lower() in FACE_FIELDS for f in altered_fields)
    has_text = any(f.lower() not in FACE_FIELDS for f in altered_fields)
    if has_face and has_text:
        return "both"
    if has_face:
        return "face"
    if has_text:
        return "text"
    return "none"


def clip_and_fraction(x, y, w, h, W, H):
    """Intersect a raw box with the image domain; return clipped corners, visible
    fraction, and a truncation flag. Raw values are never modified by the caller."""
    if None in (x, y, w, h) or W is None or H is None or w <= 0 or h <= 0:
        x2 = (x + w) if (x is not None and w is not None) else None
        y2 = (y + h) if (y is not None and h is not None) else None
        return x, y, x2, y2, None, None
    x1 = max(0, x); y1 = max(0, y)
    x2 = min(W, x + w); y2 = min(H, y + h)
    cw = max(0, x2 - x1); ch = max(0, y2 - y1)
    frac = (cw * ch) / float(w * h)
    return x1, y1, x2, y2, frac, (frac < 1.0)


# --------------------------------------------------------------------------- #
# core
# --------------------------------------------------------------------------- #
def parse_one(img_path: Path, root: Path, do_hash: bool):
    img_rel = img_path.relative_to(root)
    stem = img_path.stem
    json_path = img_path.with_suffix(".json")

    split, class_label, class_folder, attack_method, device_raw = derive_path_fields(img_rel)
    device = DEVICE_ALIAS.get(device_raw, device_raw)
    language, _ = split_stem(stem)

    w, h = image_size(img_path)  # read early so region clipping can use it

    row = {
        "image_path": img_rel.as_posix(),
        "json_path": json_path.relative_to(root).as_posix() if json_path.exists() else "",
        "split": split,
        "class_label": class_label,
        "class_folder": class_folder,
        "attack_method_folder": attack_method,
        "device_raw": device_raw,
        "device": device,
        "language": language,
        "card_stem": stem,
        "group_id": stem,
        "capture_id": f"{stem}::{device_raw}",
        "face_id": "", "face_db": "", "face_group_id": "", "gender": "",
        "json_parse_ok": False,
        "n_regions": 0,
        "n_altered_regions": 0,
        "altered_field_names": "",
        "altered_field_names_unique": "",
        "manip_type": "none",
        "img_width": w, "img_height": h,
        "resulted_cropped_width": None, "resulted_cropped_height": None,
        "original_image_width": None, "original_image_height": None,
        "has_cropping_info": False,
        "cropping_info_key": "", "crop_pipeline": "none", "crop_multiple_keys": False,
        "bbox_space": "unknown",
        "any_bbox_truncated": False,
        "homography_into_cropped": "", "original_rectangle": "",
        "image_size_bytes": img_path.stat().st_size if img_path.exists() else None,
        "image_sha256": "",
        # paired_bonafide_* filled in build() post-process
    }

    region_rows = []
    data = None
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            row["json_parse_ok"] = True
        except Exception as e:
            row["annotation_error"] = str(e)

    if data:
        pi = data.get("person_info", {}) or {}
        row["face_id"] = pi.get("face_id", "")
        row["face_db"] = pi.get("face_db", "")
        row["gender"] = pi.get("gender", "")
        row["face_group_id"] = f'{row["face_db"]}:{row["face_id"]}' if row["face_id"] else ""

        regions = data.get("regions", []) or []
        row["n_regions"] = len(regions)
        altered_fields = []
        any_trunc = False
        for r in regions:
            sa = r.get("shape_attributes", {}) or {}
            ra = r.get("region_attributes", {}) or {}
            prov = ra.get("region_provenance", "")
            fname = ra.get("field_name", "")
            x, y, ww, hh = sa.get("x"), sa.get("y"), sa.get("width"), sa.get("height")
            x1e, y1e, x2e, y2e, frac, trunc = clip_and_fraction(x, y, ww, hh, w, h)
            if trunc:
                any_trunc = True
            if prov == "altered":
                altered_fields.append(fname)
            region_rows.append({
                "image_path": img_rel.as_posix(),
                "card_stem": stem, "split": split, "class_folder": class_folder,
                "attack_method_folder": attack_method, "device_raw": device_raw,
                "device": device, "language": language,
                "field_name": fname,
                "region_provenance": prov if prov else ("original" if class_label == 0 else ""),
                "x": x, "y": y, "width": ww, "height": hh,           # RAW, unchanged
                "x1_eff": x1e, "y1_eff": y1e, "x2_eff": x2e, "y2_eff": y2e,
                "bbox_visible_fraction": frac, "bbox_truncated": trunc,
                "org_value": ra.get("org_value", ""), "new_value": ra.get("new_value", ""),
                "source": ra.get("source", ""), "target": ra.get("target", ""),
                "field_language": ra.get("language", ""),
            })
        row["n_altered_regions"] = len(altered_fields)
        row["altered_field_names"] = "|".join(altered_fields)
        row["altered_field_names_unique"] = "|".join(sorted(set(altered_fields)))
        row["manip_type"] = classify_manip(altered_fields)
        row["any_bbox_truncated"] = any_trunc

        # v3.2: discover crop metadata under ANY 'cropping_info*' key (FantasyID uses
        # 'cropping_info' and 'cropping_info-altered-recaptured' at least). Prefix-match,
        # don't hard-code; record the source key + pipeline for provenance.
        crop_keys = [k for k in data.keys() if isinstance(k, str) and k.startswith("cropping_info")]
        row["crop_multiple_keys"] = len(crop_keys) > 1
        if crop_keys:
            crop_key = "cropping_info" if "cropping_info" in crop_keys else sorted(crop_keys)[0]
            ci = data.get(crop_key, {}) or {}
        else:
            crop_key, ci = "", {}
        row["cropping_info_key"] = crop_key
        row["crop_pipeline"] = (
            "standard" if crop_key == "cropping_info"
            else "none" if crop_key == ""
            else "altered_recaptured" if "altered-recaptured" in crop_key
            else crop_key
        )
        if ci:
            row["has_cropping_info"] = True
            row["resulted_cropped_width"] = ci.get("resulted_cropped_image_width")
            row["resulted_cropped_height"] = ci.get("resulted_cropped_image_height")
            row["original_image_width"] = ci.get("original_image_width")
            row["original_image_height"] = ci.get("original_image_height")
            row["homography_into_cropped"] = json.dumps(ci.get("transformation_matrix_into_cropped"))
            row["original_rectangle"] = json.dumps(ci.get("original_rectangle_tl_tr_br_bl"))

    # bbox-space resolution (provenance of the coordinate frame; clipping above is independent)
    rc_w, rc_h = row["resulted_cropped_width"], row["resulted_cropped_height"]
    if w is not None:
        if row["has_cropping_info"] and rc_w is not None and rc_h is not None:  # v3.1: rc_h guard
            tol_w = max(CROP_TOL_MIN, CROP_TOL_FRAC * rc_w)
            tol_h = max(CROP_TOL_MIN, CROP_TOL_FRAC * rc_h)
            if w == rc_w and h == rc_h:
                row["bbox_space"] = "cropped"               # exact match
            elif abs(w - rc_w) <= tol_w and abs(h - rc_h) <= tol_h:
                row["bbox_space"] = "cropped_dim_mismatch"  # v3.1: near-match, flagged not silently lumped (no rescale)
            else:
                row["bbox_space"] = "needs_homography"
        elif not row["has_cropping_info"]:
            row["bbox_space"] = "on_disk"
    if do_hash and img_path.exists():
        row["image_sha256"] = sha256_of(img_path)

    return row, region_rows


def build(root: Path, out_dir: Path, do_hash: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    img_paths = [Path(dp) / fn for dp, _, fns in os.walk(root)
                 for fn in fns if Path(fn).suffix.lower() in IMAGE_EXTS]
    json_only = False
    if not img_paths:
        json_only = True
        img_paths = [(Path(dp) / fn).with_suffix(".jpg")
                     for dp, _, fns in os.walk(root)
                     for fn in fns if Path(fn).suffix.lower() == ".json"]

    rows, region_rows = [], []
    for p in sorted(img_paths):
        r, rr = parse_one(p, root, do_hash and not json_only)
        rows.append(r); region_rows.extend(rr)

    manifest = pd.DataFrame(rows)
    regions = pd.DataFrame(region_rows)

    # exclude anything outside train/test (e.g. examples/)
    n_excluded = int((~manifest["split"].isin(SPLITS)).sum())
    if n_excluded:
        ex = manifest.loc[~manifest["split"].isin(SPLITS), "image_path"].tolist()
        manifest = manifest[manifest["split"].isin(SPLITS)].copy()
        regions = regions[regions["image_path"].isin(set(manifest["image_path"]))].copy()
        print(f"[note] excluded {n_excluded} files outside train/test: "
              + ", ".join(ex[:8]) + (" ..." if n_excluded > 8 else ""))

    # card_in_train (leakage stratifier) — unchanged from v2
    train_stems = set(manifest.loc[manifest["split"] == "train", "card_stem"])
    manifest["card_in_train"] = (manifest["split"] == "test") & manifest["card_stem"].isin(train_stems)

    # bonafide <-> manipulated linkage
    bona = manifest[manifest["class_label"] == 0]
    bona_map = dict(zip(bona["card_stem"] + "||" + bona["device_raw"], bona["image_path"]))
    key = manifest["card_stem"] + "||" + manifest["device_raw"]
    manifest["paired_bonafide_path"] = key.map(bona_map).fillna("")
    manifest["paired_bonafide_available"] = manifest["paired_bonafide_path"] != ""

    manifest.to_csv(out_dir / "manifest.csv", index=False, lineterminator="\n")  # v3.1: stable hash
    regions.to_csv(out_dir / "regions.csv", index=False, lineterminator="\n")
    _print_report(manifest, regions, json_only)
    return manifest, regions


def _print_report(manifest, regions, json_only):
    L = "=" * 70
    print(L)
    print(f"MANIFEST: {len(manifest)} images   REGIONS: {len(regions)} annotated boxes"
          + ("   [JSON-ONLY DRY RUN]" if json_only else ""))
    print(L)
    for c in ["split", "class_folder", "attack_method_folder", "device_raw", "device",
              "language", "manip_type", "bbox_space"]:
        if c in manifest:
            print(f"\n[{c}]"); print(manifest[c].value_counts(dropna=False).to_string())

    # v3: truncation summary
    if "bbox_visible_fraction" in regions and regions["bbox_visible_fraction"].notna().any():
        tr = regions[regions["bbox_truncated"] == True]
        print("\n" + L); print("BBOX TRUNCATION (clip, don't exclude)"); print(L)
        print(f"truncated boxes: {len(tr)}  |  images with truncation: "
              f"{int(manifest['any_bbox_truncated'].sum())}")
        if len(tr):
            vf = pd.to_numeric(tr["bbox_visible_fraction"], errors="coerce")
            print(f"visible_fraction  min={vf.min():.4f}  median={vf.median():.4f}  max={vf.max():.4f}")
            print(f"boxes <70% visible: {(vf < 0.70).sum()}   <50% visible: {(vf < 0.50).sum()}"
                  "   (0 -> no exclusions needed)")
            alt = tr[tr["region_provenance"] == "altered"]
            if len(alt):
                vfa = pd.to_numeric(alt["bbox_visible_fraction"], errors="coerce")
                print(f"altered truncated boxes: {len(alt)}  min visible={vfa.min():.4f}")

    print("\n" + L); print("GROUPING / LEAKAGE DIAGNOSTICS"); print(L)
    print(f"unique card_stem: {manifest['card_stem'].nunique()}")
    print(f"paired_bonafide_available: {int(manifest['paired_bonafide_available'].sum())}"
          f" / {len(manifest)}")
    if {"train", "test"} <= set(manifest["split"].unique()):
        te = manifest[manifest["split"] == "test"]
        print("\n[test images by attack_method x card_in_train]")
        print(pd.crosstab(te["attack_method_folder"].fillna("(bonafide)"),
                          te["card_in_train"], margins=True).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", default="manifest_out")
    ap.add_argument("--no-hash", action="store_true")
    args = ap.parse_args()
    build(Path(args.root), Path(args.out_dir), do_hash=not args.no_hash)
