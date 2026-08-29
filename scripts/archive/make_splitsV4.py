#!/usr/bin/env python3
"""
make_splits.py  (v2)  --  Phase 0C: frozen, leakage-aware splits.

Reads manifest.csv and emits splits.csv + splits_meta.json.

v2 changes (re-freeze before 0D):
  * to_csv uses lineterminator="\n"        -> cross-platform-stable byte hash (was CRLF on Windows)
  * stratum renames (card_stem != template):
        known_template_d3_seen -> d3_seen_base_card
        spine_text_d3_unseen   -> spine_text_d3_unseen_card
  * new columns:
        language_in_train        -> operationalises the language/attack confound for stratified reporting
        base_card_dev_exposure   -> dev_train_seen | dev_val_seen | unseen  (sharpens the seen-d3 secondary analysis)
  * splits_meta.json now binds to its inputs: manifest_sha256, split_script_sha256,
    split_version, manifest_rows, train_cards, test_cards
  * extra invariant assertions
Grouping unit is card_stem; dev split is card-grouped, language-stratified; test untouched.
"""
import argparse, hashlib, json, random
from pathlib import Path
import pandas as pd

SPINE = {"spine_face", "spine_text", "spine_text_d3_unseen_card"}


def norm_sha256(path: Path) -> str:
    """CRLF-normalised sha256 so hashes match across Windows/Unix."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def assign_dev_roles(train_df, val_frac, seed):
    cards = train_df[["card_stem", "language"]].drop_duplicates()
    rng = random.Random(seed)
    val_cards = set()
    for lang, grp in cards.groupby("language"):
        stems = sorted(grp["card_stem"].tolist())
        rng.shuffle(stems)
        n = len(stems)
        n_val = 0 if n < 2 else max(1, round(val_frac * n))
        val_cards.update(stems[:n_val])
    role = train_df["card_stem"].apply(lambda s: "dev_val" if s in val_cards else "dev_train")
    return role, val_cards


def tag_test_stratum(row):
    m = row["attack_method_folder"] or ""
    if row["class_folder"] == "bonafide":
        return "test_bonafide"
    if m == "facedancer":
        return "spine_face"
    if m == "textdiffuserft_bfei":
        return "spine_text"
    if m == "digital_3":
        return "spine_text_d3_unseen_card" if not row["card_in_train"] else "d3_seen_base_card"
    return "other"


def main(manifest_path, out_dir, val_frac, seed):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(manifest_path, low_memory=False)
    m["attack_method_folder"] = m["attack_method_folder"].fillna("")
    if "card_in_train" not in m.columns:
        tr_stems = set(m.loc[m.split == "train", "card_stem"])
        m["card_in_train"] = (m.split == "test") & m.card_stem.isin(tr_stems)

    # dev split
    m["dev_role"] = ""
    tr = m.split == "train"
    role, val_cards = assign_dev_roles(m[tr], val_frac, seed)
    m.loc[tr, "dev_role"] = role.values

    # card -> dev exposure (for every image), and language-in-train flag
    card_role = m.loc[tr].drop_duplicates("card_stem").set_index("card_stem")["dev_role"].to_dict()
    def exposure(s):
        r = card_role.get(s)
        return {"dev_train": "dev_train_seen", "dev_val": "dev_val_seen"}.get(r, "unseen")
    m["base_card_dev_exposure"] = m["card_stem"].map(exposure)
    train_langs = set(m.loc[tr, "language"])
    m["language_in_train"] = m["language"].isin(train_langs)

    # test strata
    te = m.split == "test"
    m["dcew_stratum"] = ""
    m.loc[te, "dcew_stratum"] = m[te].apply(tag_test_stratum, axis=1)
    m["is_dcew_spine"] = m["dcew_stratum"].isin(SPINE)

    # ---- assertions (point 20) ----
    dt = set(m.loc[m.dev_role == "dev_train", "card_stem"])
    dv = set(m.loc[m.dev_role == "dev_val", "card_stem"])
    train_cards = set(m.loc[tr, "card_stem"])
    spine_cards = set(m.loc[m.is_dcew_spine, "card_stem"])
    assert dt | dv == train_cards, "dev cards must partition train cards"
    assert dt.isdisjoint(dv), "dev_train/dev_val must be card-disjoint"
    assert (m.loc[te, "dev_role"] == "").all(), "test rows must have empty dev_role"
    assert m["image_path"].is_unique, "image_path must be unique"
    assert (m.loc[m.is_dcew_spine, "class_folder"] == "attack").all(), "spine must be all attack"
    assert set(m.loc[m.is_dcew_spine, "manip_type"]).issubset({"face", "text"}), "spine must be single-type"
    a1 = dt.isdisjoint(dv)
    a2 = spine_cards.isdisjoint(train_cards)
    assert a2, "spine must be card-disjoint from all training"

    cols = ["image_path", "split", "dev_role", "base_card_dev_exposure", "card_stem", "language",
            "language_in_train", "class_label", "class_folder", "attack_method_folder",
            "manip_type", "card_in_train", "dcew_stratum", "is_dcew_spine"]
    splits = m[cols].copy()
    splits_path = out_dir / "splits.csv"
    splits.to_csv(splits_path, index=False, lineterminator="\n")   # point 18
    assert len(m) == len(splits)

    digest = norm_sha256(splits_path)
    meta = {
        "split_version": "0C-v2", "seed": seed, "val_frac": val_frac,
        "manifest_sha256": norm_sha256(manifest_path),
        "split_script_sha256": norm_sha256(__file__),
        "manifest_rows": int(len(m)),
        "train_cards": len(train_cards), "test_cards": int(m.loc[te, "card_stem"].nunique()),
        "dev_train_images": int((m.dev_role == "dev_train").sum()),
        "dev_val_images": int((m.dev_role == "dev_val").sum()),
        "dev_train_cards": len(dt), "dev_val_cards": len(dv),
        "spine_images": int(m.is_dcew_spine.sum()), "spine_cards": len(spine_cards),
        "assertion_dev_card_disjoint": a1, "assertion_spine_disjoint_from_train": a2,
        "splits_sha256": digest,
    }
    (out_dir / "splits_meta.json").write_text(json.dumps(meta, indent=2))

    L = "=" * 70
    print(L); print("PHASE 0C SPLITS v2"); print(L)
    print(f"split_version=0C-v2  seed={seed}  val_frac={val_frac}")
    print(f"manifest_sha256={meta['manifest_sha256'][:16]}...  splits_sha256={digest[:16]}...")
    print(f"\ndev_train {meta['dev_train_images']}img/{meta['dev_train_cards']}cards  "
          f"dev_val {meta['dev_val_images']}img/{meta['dev_val_cards']}cards")
    print("\n[test strata]"); print(m[te]["dcew_stratum"].value_counts().to_string())
    print(f"\nPRIMARY DCEW SPINE: {meta['spine_images']} imgs / {meta['spine_cards']} cards")
    print("[spine: stratum x language_in_train] (confound is total -> use 2 clean contrasts)")
    print(pd.crosstab(m[m.is_dcew_spine]["dcew_stratum"], m[m.is_dcew_spine]["language_in_train"]).to_string())
    print("\n[seen-d3 base_card_dev_exposure]")
    sd = m[(te) & (m.dcew_stratum == "d3_seen_base_card")]
    print(sd["base_card_dev_exposure"].value_counts().to_string())
    print("\n[integrity]  all assertions PASSED")
    print(L)
    return splits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default="splits_out")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.manifest, a.out_dir, a.val_frac, a.seed)
