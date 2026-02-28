# napari-manual-labels-editor

Manual fine-tuning toolkit for **napari Labels** (integer label masks).
Designed for “after segmentation” workflows where you need to **inspect, fix, and export** large label images safely (e.g., TIFF/LZW/BigTIFF).

Tutorial video: https://www.youtube.com/watch?v=cQf_7ExGQHk&t=15s

## What it does

This plugin operates on a **napari Labels layer** (background must be `0`) and provides a small, fast panel to:

- **Pick Active Layer**: choose the current active Image/Labels layer; if it’s an Image, it can be converted to Labels (integer).
- **Compute Stats (safe)**: compute `maxID` and `cells` without heavy `np.unique` (safer for large arrays).
- **New ID (max+1)**: set selected label to `maxID + 1` so you can paint a new object.
- **Delete selected ID**: delete the currently selected label ID (set its pixels to `0`).
- **Compact IDs Order (1..N)**: relabel present IDs to consecutive `1..N` (useful before export).
- **Merge (Shift-click)**: when merge is enabled, **Shift-click A**, then **Shift-click B** to merge **B → A** (into A).
- **Fill closed shape**: draw a closed outline, then fill the enclosed region for the selected label.
- **Filter small labels (to new layer)**: remove tiny fragments by area threshold while keeping the original layer unchanged.
- **Undo / Redo**: revert recent edits made through the plugin’s actions (pixel edits only).
- **Blink**: quick visibility toggling for spot-checking edits.
- **Jump to label (J)**: jump to a specific label ID via a dialog (sets `selected_label`).

Export:

- **Save labels to TIFF** (LZW + BigTIFF enabled).
  Use **Browse…** to pick an output path, then click **Save**.

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
4. (Optional but recommended) Click **Compute Stats (safe)**
   The status area will show `selected`, `cells`, `maxID`, and `merge` state.

### Hotkeys

- `N` — **New ID (max+1)**
- `F` — **Fill closed shape**
- `Shift+F` — **Fill closed shape + Next ID**
- `J` — **Jump to label ID** (dialog)

Notes:
- Hotkeys are captured by the napari viewer. If a key seems unresponsive, click once on the canvas to give the viewer focus.
- napari also has its own built-in tools/hotkeys (e.g., paint/fill modes); you can keep using those alongside this plugin.

### UI behavior (narrow dock)

- The dock panel can be resized narrower without hiding critical controls.
- **Browse…** and **Save** are protected from being shrunk to zero width.

### Undo / Redo behavior

- Undo/Redo reverts recent pixel edits made via the plugin’s actions (e.g., fill closed, delete, merge, relabel/filter).
- Changing the selected label / “New ID” alone does not change pixels, so it is not treated as an edit step to undo.
- Standard behavior: if you Undo and then make a new edit, the Redo history is cleared.

### Export

1. Click **Browse…** to choose an output file path (TIFF).
2. Click **Save**.

Notes:
- TIFF writer uses `compression="lzw"` and `bigtiff=True`.

## Notes

### Large LZW TIFF support

If LZW read/write fails or is slow, install `imagecodecs`:

```bash
conda install -c conda-forge imagecodecs
```

## License

BSD-3-Clause
