from __future__ import annotations

from dataclasses import dataclass


import os
import napari
import numpy as np
import tifffile as tiff
from magicgui import magicgui
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtWidgets import QLabel, QWidget


def _is_labels(layer) -> bool:
    return layer is not None and layer.__class__.__name__ == "Labels"


def _is_image(layer) -> bool:
    return layer is not None and layer.__class__.__name__ == "Image"


def _ensure_labels_layer(viewer: Viewer, layer):
    """If input is Image, convert to Labels and remove original (optional)."""
    if _is_labels(layer):
        return layer

    if _is_image(layer):
        data = np.asarray(layer.data)
        if not np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.int32, copy=False)
        else:
            data = data.astype(np.int32, copy=False)

        name_new = layer.name + "__LABELS"
        L = viewer.add_labels(data, name=name_new)
        L.opacity = 0.6
        L.mode = "paint"
        try:
            viewer.layers.remove(layer)
        except Exception:
            pass
        return L

    raise RuntimeError(f"Unsupported layer type: {layer.__class__.__name__}")


def _hover_id(layer, event) -> int:
    try:
        pos = layer.world_to_data(event.position)
        y = int(round(pos[0]))
        x = int(round(pos[1]))
        if y < 0 or x < 0:
            return 0
        arr = layer.data
        if y >= arr.shape[0] or x >= arr.shape[1]:
            return 0
        return int(arr[y, x])
    except Exception:
        return 0


def _compute_stats_chunked(arr: np.ndarray, chunk_h: int = 512):
    H, _W = arr.shape
    max_id = 0
    for y0 in range(0, H, chunk_h):
        block = np.asarray(arr[y0 : y0 + chunk_h, :])
        v = int(block.max())
        if v > max_id:
            max_id = v

    presence = np.zeros(max_id + 1, dtype=np.bool_)
    for y0 in range(0, H, chunk_h):
        block = np.asarray(arr[y0 : y0 + chunk_h, :])
        nz = block[block > 0].ravel()
        if nz.size == 0:
            continue
        bc = np.bincount(nz.astype(np.int64), minlength=max_id + 1)
        presence |= bc > 0

    cell_count = int(presence[1:].sum())
    return max_id, cell_count, presence


@dataclass
class EditorState:
    max_id: int | None = None
    cell_count: int | None = None
    hover_id: int = 0
    merge_first: int | None = None


class ManualLabelsEditor(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self.viewer = viewer
        self.state = EditorState()
        self.chunk_h = 512
        self.default_save_suffix = "_edited_LZW.tif"

        self.layer = None
        # overlay disabled (panel status only)

        # magicgui buttons

        # ---- Status panel (like mask-curator) ----
        from qtpy.QtWidgets import QSizePolicy

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Maximum
        )
        self.status_label.setStyleSheet("font-family: Menlo, monospace;")
        self._status_cache = {
            "layer": "NA",
            "hover": 0,
            "selected": 0,
            "maxID": "NA",
            "cells": "NA",
            "merge": "OFF",
        }
        self.btn_pick = self._make_btn_pick()
        self.btn_stats = self._make_btn_stats()
        self.btn_new = self._make_btn_new()
        self.btn_del = self._make_btn_del()
        self.btn_relabel = self._make_btn_relabel()
        self.merge_enabled = False
        self.btn_merge = self._make_btn_merge_toggle()

        self.btn_save = self._make_btn_save()

        # mount into this QWidget
        from qtpy.QtWidgets import QVBoxLayout

        root = QVBoxLayout(self)
        root.addWidget(QLabel("Manual Labels Editor"))
        root.addWidget(self.status_label)
        root.addWidget(self.btn_pick.native)
        root.addWidget(self.btn_stats.native)
        root.addWidget(self.btn_new.native)
        root.addWidget(self.btn_del.native)
        root.addWidget(self.btn_relabel.native)
        root.addWidget(self.btn_merge.native)
        root.addWidget(self.btn_save.native)

    # ---------- status ----------

    def _set_status(self, msg: str):
        self.viewer.status = msg
        # no overlay: status shown in dock panel
        self._status_cache["last"] = msg
        self._update_panel_status()

        # also mirror into dock status label if present
        try:
            if hasattr(self, "status_label") and self.status_label is not None:
                # keep last status line on top of panel status
                self.status_label.setText(msg)
        except Exception:
            pass

    def _pick_active_layer(self) -> None:
        """Pick the currently active layer. If Image -> convert to Labels, else keep Labels."""
        layer = getattr(
            getattr(self.viewer, "layers", None), "selection", None
        )
        layer = getattr(layer, "active", None)
        if layer is None:
            self._set_status(
                "⚠️ No active layer. Select an Image/Labels layer first."
            )
            return

        # Convert Image -> Labels (or keep Labels)
        self.layer = _ensure_labels_layer(self.viewer, layer)

        # Ensure callbacks are attached
        try:
            if self._on_mouse_move not in self.layer.mouse_move_callbacks:
                self.layer.mouse_move_callbacks.append(self._on_mouse_move)
        except Exception:
            pass

        try:
            if self._on_click_merge not in self.layer.mouse_drag_callbacks:
                self.layer.mouse_drag_callbacks.append(self._on_click_merge)
        except Exception:
            pass

        # Update panel status cache (if present)
        if hasattr(self, "_status_cache"):
            self._status_cache["layer"] = getattr(self.layer, "name", "NA")
            self._status_cache["selected"] = int(
                getattr(self.layer, "selected_label", 0) or 0
            )
            if hasattr(self, "_update_panel_status"):
                try:
                    self._update_panel_status()
                except Exception:
                    pass

        self._set_status(
            f"✅ Using layer: {self.layer.name}. Click 'Compute Stats'."
        )

    def _update_panel_status(self):
        """Update dock status panel (no mouse tracking)."""
        if self.layer is not None:
            self._status_cache["layer"] = getattr(self.layer, "name", "NA")
            self._status_cache["selected"] = int(
                getattr(self.layer, "selected_label", 0) or 0
            )
        self._status_cache["maxID"] = (
            int(self.state.max_id) if self.state.max_id is not None else "NA"
        )
        self._status_cache["cells"] = (
            int(self.state.cell_count)
            if self.state.cell_count is not None
            else "NA"
        )
        self._status_cache["merge"] = (
            "ON" if bool(getattr(self, "merge_enabled", False)) else "OFF"
        )

        s = self._status_cache
        msg = (
            f"Layer: {s['layer']}\n"
            f"selected={s['selected']} | cells={s['cells']}\n"
            f"maxID={s['maxID']} | merge={s['merge']}"
        )
        try:
            self.status_label.setText(msg)
        except Exception:
            pass

    def _on_click_merge(self, layer, event):
        if not getattr(self, "merge_enabled", False):
            return
        if event.type != "mouse_press":
            return
        if "Shift" not in event.modifiers:
            return

        hid = _hover_id(layer, event)
        if hid <= 0:
            return

        if self.state.merge_first is None:
            self.state.merge_first = hid
            return

        a = int(self.state.merge_first)
        b = int(hid)
        self.state.merge_first = None
        if a == b:
            return

        arr = self.layer.data
        with self.layer.events.data.blocker():
            arr[arr == b] = a

    # ---------- actions ----------

    def _new_id(self):
        if self.layer is None:
            return
        if self.state.max_id is None:
            return
        nid = int(self.state.max_id) + 1
        self.layer.selected_label = nid
        self._update_panel_status()

    def _delete_selected(self):

        sel = int(getattr(self.layer, "selected_label", 0) or 0)
        if self.layer is None:
            return
        if sel <= 0:
            return
        arr = self.layer.data
        with self.layer.events.data.blocker():
            arr[arr == sel] = 0
        self._update_panel_status()

    def _default_save_path(self):
        base = os.getcwd()
        stem = self.layer.name if self.layer is not None else "labels"
        try:
            p = getattr(getattr(self.layer, "source", None), "path", None)
            if p:
                base = os.path.dirname(p)
                stem = os.path.splitext(os.path.basename(p))[0]
        except Exception:
            pass
        return os.path.join(base, stem + self.default_save_suffix)

    # ---------- workers ----------
    @thread_worker
    def _worker_stats(self):
        return _compute_stats_chunked(self.layer.data, chunk_h=self.chunk_h)

    def _after_stats(self, res):
        mx, cc, _presence = res
        self.state.max_id = int(mx)
        self.state.cell_count = int(cc)

        self._update_panel_status()

    @thread_worker
    def _worker_relabel(self):
        arr = self.layer.data
        mx, _cc, presence = _compute_stats_chunked(arr, chunk_h=self.chunk_h)
        present_ids = np.nonzero(presence)[0]
        present_ids = present_ids[present_ids > 0]

        mapping = np.zeros(mx + 1, dtype=np.int32)
        mapping[present_ids] = np.arange(
            1, present_ids.size + 1, dtype=np.int32
        )

        H, _W = arr.shape
        with self.layer.events.data.blocker():
            for y0 in range(0, H, self.chunk_h):
                block = np.asarray(arr[y0 : y0 + self.chunk_h, :])
                arr[y0 : y0 + self.chunk_h, :] = mapping[block]

        new_max = int(present_ids.size)
        return new_max, new_max

    def _after_relabel(self, res):
        mx, cc = res
        self.state.max_id = mx
        self.state.cell_count = cc
        if int(getattr(self.layer, "selected_label", 0) or 0) > mx:
            self.layer.selected_label = mx

        self._update_panel_status()

    @thread_worker
    def _worker_save(self, out_path: str):
        arr = np.asarray(self.layer.data)
        import os

        if os.path.isdir(out_path):
            out_path = os.path.join(
                out_path,
                (
                    getattr(self.layer, "name", "labels")
                    + self.default_save_suffix
                ),
            )
        tiff.imwrite(
            out_path,
            arr.astype(np.int32, copy=False),
            compression="lzw",
            bigtiff=True,
        )
        return out_path

    def _after_save(self, p: str):
        self._set_status(f"💾 Saved: {p}")
        self._update_panel_status()

        # ---------- magicgui wrappers ----------
        self._update_panel_status()

    def _make_btn_pick(self):
        @magicgui(call_button="🎯 Pick Active Layer (Image or Labels)")
        def _btn():
            self._pick_active_layer()

        return _btn

    def _make_btn_stats(self):
        @magicgui(call_button="📊 Compute Stats (safe)")
        def _btn():
            if self.layer is None:
                return
            w = self._worker_stats()
            w.returned.connect(self._after_stats)
            w.start()

        return _btn

    def _make_btn_new(self):
        @magicgui(call_button="➕ New ID (max+1)")
        def _btn():
            self._new_id()

        return _btn

    def _make_btn_del(self):
        @magicgui(call_button="❌ Delete selected ID")
        def _btn():
            self._delete_selected()

        return _btn

    def _make_btn_relabel(self):
        @magicgui(call_button="≡ Compact IDs Order (1..N)")
        def _btn():
            if self.layer is None:
                return
            w = self._worker_relabel()
            w.returned.connect(self._after_relabel)
            w.start()

        return _btn

    def _set_merge_enabled(self, enabled: bool) -> None:
        """Enable/disable merge action (Shift-click)."""
        self.merge_enabled = bool(enabled)
        label = f"Merge: {'ON' if self.merge_enabled else 'OFF'}"

        # update magicgui button label
        try:
            self.btn_merge.text = label
        except Exception:
            pass

        # update underlying Qt button label (more reliable)
        try:
            self.btn_merge.native.setText(label)
        except Exception:
            pass

    def _toggle_merge(self) -> None:
        self.merge_enabled = not bool(getattr(self, "merge_enabled", False))
        # update structured status if exists
        try:
            self._status_cache["merge"] = "ON" if self.merge_enabled else "OFF"
            self._update_panel_status()
        except Exception:
            pass

    def _make_btn_merge_toggle(self):
        @magicgui(call_button="Merge: Shift+A+B → A")
        def _btn():
            self._toggle_merge()

        return _btn

    def _make_btn_save(self):

        @magicgui(
            out_path={"label": "Save path"},
            call_button="💾 Save labels (LZW)",
        )
        def _btn(out_path: str = ""):

            if self.layer is None:

                self._set_status("⚠️ Pick a layer first.")

                return

            if not out_path:

                out_path = self._default_save_path()

            # allow directory path: auto-append filename

            import os

            if os.path.isdir(out_path):

                out_path = os.path.join(
                    out_path,
                    (
                        getattr(self.layer, "name", "labels")
                        + self.default_save_suffix
                    ),
                )

            self._set_status(f"⏳ Saving to {out_path} ...")

            w = self._worker_save(out_path)

            w.returned.connect(self._after_save)

            w.start()

            # panel refresh

            try:

                self._update_panel_status()

            except Exception:

                pass

        return _btn


# -------------------------
# napari widget factory (MUST be top-level)
# -------------------------


def make_manual_labels_editor_widget(viewer: Viewer | None = None) -> QWidget:
    """Napari widget factory (npe2)."""
    if viewer is None:
        try:
            viewer = napari.current_viewer()
        except Exception as e:
            raise RuntimeError(
                "No active napari viewer found. Open napari first."
            ) from e
    return ManualLabelsEditor(viewer)
