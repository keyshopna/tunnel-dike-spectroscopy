"""
sed_rename_clean.py

Workflow for ASD/NaturaSpec .sed hyperspectral files:
  1. INSPECT   - peek at header fields of one file so you know which
                 field holds your sample name (Comment, File Name, etc.)
  2. RENAME    - batch-rename .sed + matching .raw pairs using a chosen
                 header field, so filenames match your sample IDs
  3. CLEAN     - strip noisy edge bands, fix detector-splice jumps,
                 export tidy CSVs (wavelength, reflectance) per sample

Usage:
    python sed_rename_clean.py inspect path/to/one_file.sed
    python sed_rename_clean.py rename  path/to/folder --field Comment
    python sed_rename_clean.py clean   path/to/folder --outdir cleaned_csv

Only standard library + pandas/numpy needed.
"""

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Core parsing: split a .sed file into (header_dict, data_dataframe)
# ----------------------------------------------------------------------
def parse_sed(filepath: Path):
    """
    Reads a .sed file and returns:
      header: dict of metadata key -> value (as found before the data table)
      data:   DataFrame with columns like ['Wvl', 'Rad', 'Ref%'] (varies by file)
    """
    header = {}
    data_start_idx = None

    with open(filepath, "r", errors="ignore") as f:
        lines = f.readlines()

    # Find where the numeric data table begins: the first line whose
    # first token is a float AND the line before it looks like a
    # column-header row (contains 'Wvl' or similar), OR just the first
    # purely-numeric line as a fallback.
    col_header_idx = None
    for i, line in enumerate(lines):
        if re.search(r"\bWvl\b", line, re.IGNORECASE):
            col_header_idx = i
            data_start_idx = i + 1
            break

    if data_start_idx is None:
        # Fallback: first line that starts with a number
        for i, line in enumerate(lines):
            tok = line.strip().split()
            if tok and re.match(r"^-?\d+(\.\d+)?$", tok[0]):
                data_start_idx = i
                break

    # Parse header lines (everything before the data table) as Key: Value
    header_lines = lines[:data_start_idx] if data_start_idx else lines
    for line in header_lines:
        if ":" in line:
            key, _, val = line.partition(":")
            key, val = key.strip(), val.strip()
            if key:
                header[key] = val

    # Parse the data table
    data = None
    if data_start_idx is not None:
        col_names = None
        if col_header_idx is not None:
            col_names = re.split(r"\t+|\s{2,}", lines[col_header_idx].strip())

        rows = []
        for line in lines[data_start_idx:]:
            parts = re.split(r"\t+|\s+", line.strip())
            if len(parts) < 2:
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue  # skip stray non-numeric lines

        if rows:
            width = len(rows[0])
            if not col_names or len(col_names) != width:
                col_names = [f"col{i}" for i in range(width)]
            data = pd.DataFrame(rows, columns=col_names)

    return header, data


# ----------------------------------------------------------------------
# 1. INSPECT
# ----------------------------------------------------------------------
def cmd_inspect(args):
    filepath = Path(args.file)
    header, data = parse_sed(filepath)

    print(f"\n--- Header fields found in {filepath.name} ---")
    for k, v in header.items():
        print(f"  {k!r}: {v!r}")

    if data is not None:
        print(f"\n--- Data table preview ({data.shape[0]} rows, columns: {list(data.columns)}) ---")
        print(data.head())
    else:
        print("\n[!] Could not locate a numeric data table — check file format manually.")

    print(
        "\nPick the header field that holds your sample ID "
        "(commonly 'Comment' or 'File Name') and use it with the 'rename' command."
    )


# ----------------------------------------------------------------------
# 2. RENAME
# ----------------------------------------------------------------------
def safe_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w\-.]+", "_", s)  # replace spaces/illegal chars
    return s or "unnamed"


def cmd_rename(args):
    folder = Path(args.folder)
    field = args.field
    dry_run = args.dry_run

    sed_files = sorted(folder.glob("*.sed"))
    if not sed_files:
        print(f"No .sed files found in {folder}")
        return

    seen_names = {}
    for sed_path in sed_files:
        header, _ = parse_sed(sed_path)
        raw_name = header.get(field)

        if not raw_name:
            print(f"[skip] {sed_path.name}: field '{field}' not found in header")
            continue

        new_base = safe_filename(raw_name)

        # avoid collisions if multiple files share the same field value
        count = seen_names.get(new_base, 0)
        seen_names[new_base] = count + 1
        if count > 0:
            new_base = f"{new_base}_{count+1}"

        raw_path = sed_path.with_suffix(".raw")  # matching binary file, if present

        new_sed_path = sed_path.with_name(new_base + ".sed")
        print(f"{sed_path.name}  ->  {new_sed_path.name}")

        if not dry_run:
            shutil.move(str(sed_path), str(new_sed_path))

        if raw_path.exists():
            new_raw_path = raw_path.with_name(new_base + ".raw")
            print(f"{raw_path.name}  ->  {new_raw_path.name}")
            if not dry_run:
                shutil.move(str(raw_path), str(new_raw_path))
        else:
            print(f"  [note] no matching .raw file for {sed_path.stem}")

    if dry_run:
        print("\n(dry run — no files were actually renamed; drop --dry-run to apply)")


# ----------------------------------------------------------------------
# 3. CLEAN
# ----------------------------------------------------------------------
def clean_spectrum(df: pd.DataFrame, wvl_col="Wvl", value_col=None,
                    trim_low=400, trim_high=2450,
                    splice_regions=((1000, 1010), (1790, 1810))):
    """
    Basic cleaning:
      - trims edge bands below trim_low / above trim_high (noisiest regions)
      - flags/interpolates across detector-splice jump regions
    Adjust trim_low/trim_high/splice_regions to match your instrument's
    known noisy bands (check your NaturaSpec Plus manual for exact values).
    """
    if value_col is None:
        # guess the reflectance-like column (contains 'Ref' or is the 2nd col)
        candidates = [c for c in df.columns if "ref" in c.lower()]
        value_col = candidates[0] if candidates else df.columns[1]

    out = df[[wvl_col, value_col]].copy()
    out.columns = ["wavelength", "reflectance"]

    out = out[(out["wavelength"] >= trim_low) & (out["wavelength"] <= trim_high)]

    for lo, hi in splice_regions:
        mask = (out["wavelength"] >= lo) & (out["wavelength"] <= hi)
        out.loc[mask, "reflectance"] = np.nan
    out["reflectance"] = out["reflectance"].interpolate(method="linear")

    return out.reset_index(drop=True)


def cmd_clean(args):
    folder = Path(args.folder)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sed_files = sorted(folder.glob("*.sed"))
    if not sed_files:
        print(f"No .sed files found in {folder}")
        return

    for sed_path in sed_files:
        header, data = parse_sed(sed_path)
        if data is None:
            print(f"[skip] {sed_path.name}: no data table parsed")
            continue

        wvl_col = next((c for c in data.columns if "wvl" in c.lower()), data.columns[0])
        cleaned = clean_spectrum(data, wvl_col=wvl_col)

        out_path = outdir / f"{sed_path.stem}_clean.csv"
        cleaned.to_csv(out_path, index=False)
        print(f"{sed_path.name} -> {out_path.name}  ({len(cleaned)} bands)")

    print(f"\nDone. Cleaned CSVs in: {outdir}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="Preview one .sed file's header + data")
    p_inspect.add_argument("file")
    p_inspect.set_defaults(func=cmd_inspect)

    p_rename = sub.add_parser("rename", help="Batch rename .sed/.raw pairs from a header field")
    p_rename.add_argument("folder")
    p_rename.add_argument("--field", default="Comment", help="Header field to use as new filename (default: Comment)")
    p_rename.add_argument("--dry-run", action="store_true", help="Preview renames without applying them")
    p_rename.set_defaults(func=cmd_rename)

    p_clean = sub.add_parser("clean", help="Clean spectra and export CSVs")
    p_clean.add_argument("folder")
    p_clean.add_argument("--outdir", default="cleaned_csv")
    p_clean.set_defaults(func=cmd_clean)

    args = parser.parse_args()
    args.func(args)
