"""Extrait les langues parlées depuis le SNAP brut Pokec, joint sur user_id.

Le subset FairGNN ne garde que 8 langues (anglicky, nemecky, rusky, francuzsky,
spanielsky, taliansky, slovensky, japonsky). Le SNAP brut contient le champ
``spoken_languages`` en **texte libre** — donc des langues comme **madarsky**
(hongrois), **cesky** (tchèque), **polsky** (polonais) y sont. C'est cette
information qu'on essaie de récupérer pour avoir des proxys ethniques /
linguistiques plus riches que le binaire `region`.

Le SNAP profiles est ~1.7 GB → streaming line-by-line, jamais chargé en mémoire.
On garde seulement les 66k users du subset (filtrage on-the-fly via un set).

Output : ``results/metrics/snap_languages_pokec_subset.csv`` avec une ligne
par (user_id, langue) trouvée. Plus un summary par langue avec gender/region.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
SNAP_PROFILES = ROOT / "data" / "raw" / "pokec-z" / "soc-pokec-profiles.txt"
SUBSET_CSV = ROOT / "data" / "raw" / "pokec-z" / "region_job_2.csv"
OUT_DIR = ROOT / "results" / "metrics"
# Per-user binary CSV joinable to the FairGNN subset, used as new sensitive axes.
MINORITY_CSV = ROOT / "data" / "raw" / "pokec-z" / "minority_speakers.csv"

# Token sets for the 3 minority axes we want to expose as sensitive attributes.
# Each set covers the most common Slovak spellings (with and without diacritics).
HUNGARIAN_TOKENS = {"madarsky", "madarcina", "magyar"}
ROMA_TOKENS = {"cigansky", "ciganstina", "romsky", "romcin", "romanes", "ciganski", "romski"}
SIGN_TOKENS = {"posunkovu", "posunkova", "posunkovy", "znakovy", "znakova"}

# Column index of spoken_languages in the raw SNAP TSV (1-indexed: col 11 → idx 10).
COL_USER_ID = 0
COL_GENDER = 3
COL_REGION = 4
COL_AGE = 7
COL_LANGUAGES = 10

# Slovak word for "(language)X" — we'll lowercase and strip punctuation.
# Most are *-cky / *-sky pattern.
KNOWN_LANGUAGE_TOKENS = {
    "anglicky": "english",
    "nemecky": "german",
    "rusky": "russian",
    "francuzsky": "french",
    "spanielsky": "spanish",
    "taliansky": "italian",
    "slovensky": "slovak",
    "japonsky": "japanese",
    # Likely-present-in-raw-but-curated-out :
    "madarsky": "hungarian",
    "cesky": "czech",
    "polsky": "polish",
    "ukrajinsky": "ukrainian",
    "rumunsky": "romanian",
    "chorvatsky": "croatian",
    "srbsky": "serbian",
    "portugalsky": "portuguese",
    "holandsky": "dutch",
    "svedsky": "swedish",
    "cinsky": "chinese",
    "kórejsky": "korean",
    "korejsky": "korean",
    "arabsky": "arabic",
    "hebrejsky": "hebrew",
    "latincina": "latin",
    "esperanto": "esperanto",
    "romsky": "romani",
    "ciganstina": "romani",
}

TOKEN_RE = re.compile(r"[a-záäčďéíĺľňóôŕšťúýž]+", re.IGNORECASE)


def load_subset_user_ids(csv_path: Path) -> tuple[set[int], pl.DataFrame]:
    """Return (set of user_ids, polars df with user_id+gender+region+AGE)."""
    df = pl.read_csv(csv_path, columns=["user_id", "gender", "region", "AGE"]).with_columns(
        pl.col("user_id").cast(pl.Int64)
    )
    ids = set(df["user_id"].to_list())
    return ids, df


def stream_parse_snap(snap_path: Path, target_ids: set[int]) -> dict[int, list[str]]:
    """Stream the SNAP file, return ``{user_id: [normalized language tokens]}``.

    Only target_ids are kept. ~1.7 GB read once line-by-line, no full load.
    """
    user_to_tokens: dict[int, list[str]] = {}
    n_seen = 0
    n_kept = 0
    with snap_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            n_seen += 1
            parts = line.split("\t")
            if len(parts) <= COL_LANGUAGES:
                continue
            try:
                uid = int(parts[COL_USER_ID])
            except ValueError:
                continue
            if uid not in target_ids:
                continue
            n_kept += 1

            raw = parts[COL_LANGUAGES].strip().lower()
            if raw in ("", "null"):
                user_to_tokens[uid] = []
                continue
            tokens = TOKEN_RE.findall(raw)
            user_to_tokens[uid] = tokens

            if n_kept % 10_000 == 0:
                print(f"  parsed {n_seen:>9,} lines, kept {n_kept:>6,}", flush=True)

    print(f"\nSNAP parse done : {n_seen:,} lines, {n_kept:,} subset users matched")
    return user_to_tokens


def aggregate_prevalence(
    user_to_tokens: dict[int, list[str]],
    subset_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build (per-language prevalence, per-language gender/region gap).

    Vectorised join on user_id ; no Python loops over rows.
    """
    rows: list[dict] = []
    for uid, tokens in user_to_tokens.items():
        for tok in tokens:
            rows.append({"user_id": uid, "language_token": tok})
    long = pl.DataFrame(rows)
    if long.height == 0:
        return long, long

    # Join with subset metadata
    long = long.join(subset_df, on="user_id", how="left")

    # Tokens that appear at least 50 times = candidate languages
    counts = long.group_by("language_token").len().sort("len", descending=True)

    # Mark known languages (vs noise from free-text)
    known_tokens = pl.DataFrame(
        {
            "language_token": list(KNOWN_LANGUAGE_TOKENS.keys()),
            "english_name": list(KNOWN_LANGUAGE_TOKENS.values()),
        }
    )
    counts = counts.join(known_tokens, on="language_token", how="left").rename({"len": "count"})

    # For known tokens : gender / region prevalence
    n_total = subset_df.height
    n_g0 = subset_df.filter(pl.col("gender") == 0).height
    n_g1 = subset_df.filter(pl.col("gender") == 1).height
    n_r0 = subset_df.filter(pl.col("region") == 0).height
    n_r1 = subset_df.filter(pl.col("region") == 1).height

    per_lang_gender = (
        long.filter(pl.col("language_token").is_in(list(KNOWN_LANGUAGE_TOKENS.keys())))
        .group_by("language_token")
        .agg(
            (pl.col("gender") == 0).sum().alias("n_g0"),
            (pl.col("gender") == 1).sum().alias("n_g1"),
            (pl.col("region") == 0).sum().alias("n_r0"),
            (pl.col("region") == 1).sum().alias("n_r1"),
            pl.col("user_id").n_unique().alias("n_users"),
        )
        .with_columns(
            (pl.col("n_users") / n_total * 100).alias("prevalence_pct"),
            (pl.col("n_g0") / n_g0 * 100).alias("prev_g0_pct"),
            (pl.col("n_g1") / n_g1 * 100).alias("prev_g1_pct"),
            (pl.col("n_r0") / n_r0 * 100).alias("prev_r0_pct"),
            (pl.col("n_r1") / n_r1 * 100).alias("prev_r1_pct"),
        )
        .with_columns(
            (pl.col("prev_g1_pct") - pl.col("prev_g0_pct")).alias("gender_gap_pp"),
            (pl.col("prev_r1_pct") - pl.col("prev_r0_pct")).alias("region_gap_pp"),
        )
        .join(known_tokens, on="language_token", how="left")
        .sort("prevalence_pct", descending=True)
    )

    return counts, per_lang_gender


def build_minority_speakers_df(
    user_to_tokens: dict[int, list[str]],
    subset_df: pl.DataFrame,
) -> pl.DataFrame:
    """Per-user wide table with binary `hungarian`, `roma`, `sign` columns.

    Aligned to ``subset_df`` user_ids — users absent from SNAP get 0 on every
    axis (i.e. "not declared a minority language" rather than NA). The output
    is meant to be joined directly into the FairGNN feature CSV at load time.
    """
    rows = []
    for uid, tokens in user_to_tokens.items():
        toks = set(tokens)
        rows.append(
            {
                "user_id": uid,
                "hungarian": int(bool(toks & HUNGARIAN_TOKENS)),
                "roma": int(bool(toks & ROMA_TOKENS)),
                "sign": int(bool(toks & SIGN_TOKENS)),
            }
        )
    if not rows:
        # Should never happen on real data, but keep the shape valid.
        return subset_df.select("user_id").with_columns(
            pl.lit(0, dtype=pl.Int64).alias("hungarian"),
            pl.lit(0, dtype=pl.Int64).alias("roma"),
            pl.lit(0, dtype=pl.Int64).alias("sign"),
        )
    raw = pl.DataFrame(rows).with_columns(pl.col("user_id").cast(pl.Int64))
    # Left-join onto subset so every subset user has a row, fill missing with 0.
    return (
        subset_df.select("user_id")
        .join(raw, on="user_id", how="left")
        .with_columns(
            pl.col("hungarian").fill_null(0).cast(pl.Int64),
            pl.col("roma").fill_null(0).cast(pl.Int64),
            pl.col("sign").fill_null(0).cast(pl.Int64),
        )
    )


def main() -> None:
    if not SNAP_PROFILES.exists():
        print(f"[FATAL] SNAP profiles missing: {SNAP_PROFILES}", file=sys.stderr)
        sys.exit(1)
    if not SUBSET_CSV.exists():
        print(f"[FATAL] subset csv missing: {SUBSET_CSV}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading subset user_ids from {SUBSET_CSV}")
    target_ids, subset_df = load_subset_user_ids(SUBSET_CSV)
    print(f"  → {len(target_ids):,} users in subset")

    print(f"\nStreaming {SNAP_PROFILES} ({SNAP_PROFILES.stat().st_size / 1e9:.2f} GB)")
    user_to_tokens = stream_parse_snap(SNAP_PROFILES, target_ids)

    print("\nAggregating prevalence...")
    counts_df, per_lang_df = aggregate_prevalence(user_to_tokens, subset_df)

    raw_counts_path = OUT_DIR / "snap_language_token_counts.csv"
    per_lang_path = OUT_DIR / "snap_languages_per_lang_gender_region.csv"
    counts_df.write_csv(raw_counts_path)
    per_lang_df.write_csv(per_lang_path)

    # Per-user binary CSV — the actual artefact consumed by run_minority_fairness.py
    minority_df = build_minority_speakers_df(user_to_tokens, subset_df)
    minority_df.write_csv(MINORITY_CSV)

    print("\n=== Top 30 tokens (any string in spoken_languages) ===")
    with pl.Config(tbl_rows=30, fmt_str_lengths=30):
        print(counts_df.head(30))

    print("\n=== Per-langue prévalence + gender gap + region gap ===")
    with pl.Config(tbl_rows=30, tbl_cols=12, fmt_str_lengths=20, float_precision=2):
        print(per_lang_df)

    print("\n=== Minority speakers per axis (joined to subset) ===")
    print(
        minority_df.select(
            pl.col("hungarian").sum().alias("n_hungarian"),
            pl.col("roma").sum().alias("n_roma"),
            pl.col("sign").sum().alias("n_sign"),
            pl.len().alias("n_total"),
        )
    )

    print("\nWrote :")
    print(f"  {raw_counts_path}")
    print(f"  {per_lang_path}")
    print(f"  {MINORITY_CSV}")


if __name__ == "__main__":
    main()
