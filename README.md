# napari-manual-labels-editor

Manual fine-tuning toolkit for **napari Labels** (integer label masks).
Designed for “after segmentation” workflows where you need to **inspect, fix, and export** large label images safely (e.g., TIFF/LZW/BigTIFF).

## What it does

This plugin operates on a **napari Labels layer** (background must be `0`) and provides a small, fast panel to:

- **Pick Active Layer**: choose the current active Image/Labels layer; if it’s an Image, it can be converted to Labels (integer).
- **Compute Stats (safe)**: compute `maxID` and `cells` without heavy `np.unique` (safer for large arrays).
- **New ID (max+1)**: set selected label to `maxID + 1` so you can paint a new object.
- **Delete selected ID**: delete the currently selected label ID (set its pixels to `0`).
- **Compact IDs Order (1..N)**: relabel present IDs to consecutive `1..N` (useful before export).
- **Merge (Shift-click)**: when merge is enabled, **Shift-click A**, then **Shift-click B** to merge **B → A** (into A).

Export:

- **Save labels to TIFF (LZW)** (BigTIFF enabled).
  If you provide a directory as “Save path”, the plugin auto-appends a filename.

## Compatibility

- Python: 3.10–3.13 (tested on 3.11)
- napari: 0.6.x
- Layer type: **Labels** (integer), background `0`

## Installation

### From PyPI (recommended)

```bash
pip install napari-manual-labels-editor
```

If you don’t have napari yet:

```bash
pip install "napari[all]" napari-manual-labels-editor
```

### From GitHub (latest main)

```bash
pip install -U "git+https://github.com/yaohualee1215-bit/napari-manual-labels-editor.git"
```

## Usage (inside napari)

1. Launch napari and load your **Labels** layer (and optional background image).
2. Open the plugin panel:

   **Plugins → Manual Labels Editor → Manual Labels Editor**

3. Click **Pick Active Layer (Image or Labels)**
   Make sure the layer you want is the **active** layer in the layer list first.
4. Click **Compute Stats (safe)**
   The status area will show `selected`, `cells`, `maxID`, and `merge` state.

### Editing actions

- **New ID (max+1)** → then paint to add a new label
- **Delete selected ID** → remove a label
- **Compact IDs Order (1..N)** → relabel to `1..N`
- **Merge (Shift-click)** → enable merge, then Shift-click label **A** then label **B** to merge **B → A**

### Export

1. Set **Save path** (file path or directory)
2. Click **Save labels to TIFF (LZW)**

Notes:
- TIFF writer uses `compression="lzw"` and `bigtiff=True`.
- If “Save path” is a directory, output becomes:
  `<dir>/<layer_name>_edited_LZW.tif` (or your configured suffix).

## Notes

### Large LZW TIFF support

If LZW read/write fails or is slow, install `imagecodecs`:

```bash
conda install -c conda-forge imagecodecs
```

## Publish to PyPI (maintainers)

Maintainer-only. Regular users should install from PyPI above.

### Release workflow (one block)

```bash
# from repo root
# 1) bump version in pyproject.toml (e.g., 0.1.0 -> 0.1.1)

# 2) run checks
pre-commit run --all-files

# 3) build + upload
python -m pip install -U build twine
rm -rf dist build *.egg-info
python -m build
twine check dist/*
twine upload dist/*

# 4) tag on GitHub
git tag v0.1.1
git push --tags
```

### Clean env smoke test (optional)

```bash
conda create -n napari-mlabels-test -y python=3.11
conda activate napari-mlabels-test
python -m pip install -U pip
python -m pip install napari-manual-labels-editor

python - <<'PY'
import importlib.metadata as im
print("pkg version:", im.version("napari-manual-labels-editor"))
eps = [e for e in im.entry_points(group="napari.manifest") if "manual-labels-editor" in e.name]
print("entrypoints:", eps)
PY

napari
```

## License

BSD-3-Clause
