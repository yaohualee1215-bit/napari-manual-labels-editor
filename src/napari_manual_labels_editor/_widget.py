from __future__ import annotations

from dataclasses import dataclass
from collections import deque


import os
import napari
from qtpy.QtCore import QTimer
import numpy as np
from scipy import ndimage as ndi
import tifffile as tiff
from magicgui import magicgui
from napari.qt.threading import thread_worker
from napari.viewer import Viewer
from qtpy.QtWidgets import QLabel, QWidget, QPushButton, QHBoxLayout


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
        self.btn_fill_closed = self._make_btn_fill_closed()
        self.btn_filter_area = self._make_btn_filter_area()

        # ---- Undo / Redo (memory-friendly local history) ----
        self._history_limit = 10
        self._undo_stack = deque(maxlen=self._history_limit)
        self._redo_stack = deque(maxlen=self._history_limit)
        self.btn_undo = QPushButton("Undo")
        self.btn_redo = QPushButton("Redo")
        # ---- Blink / refresh helpers ----
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._blink_toggle)
        self._blink_interval_ms = 500
        self._blink_on_ms = 500
        self._blink_off_ms = 500
        self._blink_timer.setInterval(self._blink_interval_ms)
        self._blink_running = False
        self._blink_target_visible = True

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
        root.addWidget(self.btn_fill_closed.native)
        root.addWidget(self.btn_filter_area.native)

        undo_row = QHBoxLayout()
        undo_row.addWidget(self.btn_undo)
        undo_row.addWidget(self.btn_redo)
        root.addLayout(undo_row)
        # ---- Blink controls ----
        self.btn_blink_toggle = QPushButton("Blink OFF")
        self.btn_blink_05 = QPushButton("0.5s")
        self.btn_blink_1 = QPushButton("1s")
        self.btn_blink_2 = QPushButton("2s")

        blink_row = QHBoxLayout()
        blink_row.addWidget(self.btn_blink_toggle)
        blink_row.addWidget(self.btn_blink_05)
        blink_row.addWidget(self.btn_blink_1)
        blink_row.addWidget(self.btn_blink_2)
        root.addLayout(blink_row)

        self.btn_blink_toggle.clicked.connect(self._toggle_blink_running)
        self.btn_undo.clicked.connect(self._undo_last)
        self.btn_redo.clicked.connect(self._redo_last)
        self.btn_blink_05.clicked.connect(
            lambda: self._set_blink_interval_ms(500)
        )
        self.btn_blink_1.clicked.connect(
            lambda: self._set_blink_interval_ms(1000)
        )
        self.btn_blink_2.clicked.connect(
            lambda: self._set_blink_interval_ms(2000)
        )

        # ---- Keyboard shortcuts ----
        # self._bind_shortcuts()  # disabled to avoid accidental triggers

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

        # auto-sync stats so New ID / next step works without Compute Stats
        try:
            self.state.max_id = int(np.max(self.layer.data))
        except Exception:
            self.state.max_id = None
        self._update_panel_status()
        self._bind_layer_hotkeys()
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

        # history record (memory-friendly): only store changed pixels for label b -> a
        yy, xx = np.where(arr == b)
        if yy.size == 0:
            return
        before_vals = arr[yy, xx].copy()
        after_vals = np.full_like(before_vals, a)
        self._push_history(
            {
                "op": f"merge {b}->{a}",
                "yy": yy,
                "xx": xx,
                "before_vals": before_vals,
                "after_vals": after_vals,
            }
        )

        with self.layer.events.data.blocker():
            arr[arr == b] = a
        self._refresh_labels_layer(pulse=True)

    def _push_history(self, rec: dict):
        """Push one local edit record to undo stack and clear redo."""
        try:
            self._undo_stack.append(rec)
            self._redo_stack.clear()
        except (AttributeError, RuntimeError, TypeError):
            pass

    def _apply_history_record(self, rec: dict, reverse: bool = True):
        """Apply one history record. reverse=True means undo, else redo."""
        if self.layer is None or not rec:
            return
        arr = self.layer.data
        yy = rec.get("yy", None)
        xx = rec.get("xx", None)
        before_vals = rec.get("before_vals", None)
        after_vals = rec.get("after_vals", None)
        if (
            yy is None
            or xx is None
            or before_vals is None
            or after_vals is None
        ):
            return

        vals = before_vals if reverse else after_vals
        with self.layer.events.data.blocker():
            arr[yy, xx] = vals

        self._refresh_labels_layer(pulse=True)
        self._update_panel_status()

    def _undo_last(self):
        try:
            if len(self._undo_stack) == 0:
                self._set_status("Nothing to undo.")
                return
            rec = self._undo_stack.pop()
            self._apply_history_record(rec, reverse=True)
            self._redo_stack.append(rec)
            self._set_status(f"↶ Undo: {rec.get('op', 'edit')}")
        except (AttributeError, RuntimeError, TypeError) as e:
            self._set_status(f"Undo failed: {e}")

    def _redo_last(self):
        try:
            if len(self._redo_stack) == 0:
                self._set_status("Nothing to redo.")
                return
            rec = self._redo_stack.pop()
            self._apply_history_record(rec, reverse=False)
            self._undo_stack.append(rec)
            self._set_status(f"↷ Redo: {rec.get('op', 'edit')}")
        except (AttributeError, RuntimeError, TypeError) as e:
            self._set_status(f"Redo failed: {e}")

    # ---------- actions ----------

    def _new_id(self):
        if self.layer is None:
            return
        if self.state.max_id is None:
            self.state.max_id = int(np.max(self.layer.data))

        nid = int(np.max(self.layer.data)) + 1
        self.layer.selected_label = nid
        self.state.max_id = nid
        self._update_panel_status()

    def _delete_selected(self):

        sel = int(getattr(self.layer, "selected_label", 0) or 0)
        if self.layer is None:
            return
        if sel <= 0:
            return
        arr = self.layer.data

        # history record (memory-friendly): only store changed pixels for label sel -> 0
        yy, xx = np.where(arr == sel)
        if yy.size == 0:
            return
        before_vals = arr[yy, xx].copy()
        after_vals = np.zeros_like(before_vals)
        self._push_history(
            {
                "op": f"delete {sel}",
                "yy": yy,
                "xx": xx,
                "before_vals": before_vals,
                "after_vals": after_vals,
            }
        )

        with self.layer.events.data.blocker():
            arr[arr == sel] = 0
        self._update_panel_status()
        self._refresh_labels_layer(pulse=True)

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
        @magicgui(call_button="Pick Active Layer (Image or Labels)")
        def _btn():
            self._pick_active_layer()

        return _btn

    def _make_btn_stats(self):
        @magicgui(call_button="Compute Stats (safe)")
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
            call_button="Save labels (LZW)",
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

    # ---------- refresh / blink ----------
    def _toggle_layer_visibility_once(self):
        """Force a quick visibility toggle as a refresh fallback."""
        if self.layer is None:
            return
        try:
            v = bool(getattr(self.layer, "visible", True))
            self.layer.visible = not v
            self.layer.visible = v
        except (AttributeError, RuntimeError, TypeError):
            pass

    def _refresh_labels_layer(self, pulse: bool = False):
        """Refresh labels layer display after edit operations."""
        if self.layer is None:
            return
        try:
            # Most cases this is enough
            self.layer.refresh()
        except (AttributeError, RuntimeError, TypeError):
            pass

        # napari sometimes needs a visible toggle to repaint labels immediately
        self._toggle_layer_visibility_once()

        if pulse:
            self._pulse_once()

        try:
            self._update_panel_status()
        except (AttributeError, RuntimeError, TypeError):
            pass

    def _blink_toggle(self):
        if self.layer is None:
            return
        try:
            now_vis = bool(getattr(self.layer, "visible", True))
            # 切换状态
            self.layer.visible = not now_vis
            # 切换后决定下一次等待多久：
            # 切换后可见 => 接下来保持“亮屏”时长
            # 切换后不可见 => 接下来保持“黑屏”时长（固定0.5s）
            after_vis = bool(getattr(self.layer, "visible", True))
            next_ms = self._blink_on_ms if after_vis else self._blink_off_ms
            self._blink_timer.setInterval(int(next_ms))
        except (AttributeError, RuntimeError, TypeError):
            self._stop_blink()

    def _set_blink_interval_ms(self, ms: int):
        # 用户按钮设置“亮屏持续时间”；黑屏固定 0.5s
        self._blink_on_ms = int(ms)
        self._blink_off_ms = 500
        self._blink_interval_ms = self._blink_on_ms  # 兼容旧状态文本/变量
        try:
            # 如果当前正在亮着，则下一跳按亮屏时长；若当前是黑屏，则按黑屏时长
            cur_vis = (
                bool(getattr(self.layer, "visible", True))
                if self.layer is not None
                else True
            )
            self._blink_timer.setInterval(
                self._blink_on_ms if cur_vis else self._blink_off_ms
            )
        except (AttributeError, RuntimeError, TypeError):
            pass
        try:
            self._set_status(
                f"Blink pattern = ON {self._blink_on_ms/1000:.1f}s / OFF {self._blink_off_ms/1000:.1f}s"
            )
        except (AttributeError, RuntimeError, TypeError):
            pass

    def _start_blink(self):
        if self.layer is None:
            self._set_status("⚠️ Pick a layer first.")
            return
        self._blink_running = True
        self._blink_target_visible = True
        try:
            self.layer.visible = True  # 从亮屏开始
        except (AttributeError, RuntimeError, TypeError):
            pass
        try:
            self._blink_timer.setInterval(int(self._blink_on_ms))  # 先亮这么久
        except (AttributeError, RuntimeError, TypeError):
            pass
        self._blink_timer.start()
        if hasattr(self, "btn_blink_toggle"):
            self.btn_blink_toggle.setText("Blink ON")
        self._set_status(
            f"Blink ON (ON {self._blink_on_ms/1000:.1f}s / OFF {self._blink_off_ms/1000:.1f}s)"
        )

    def _stop_blink(self):
        self._blink_running = False
        try:
            self._blink_timer.stop()
        except (AttributeError, RuntimeError, TypeError):
            pass
        try:
            if self.layer is not None:
                self.layer.visible = True
        except (AttributeError, RuntimeError, TypeError):
            pass
        if hasattr(self, "btn_blink_toggle"):
            self.btn_blink_toggle.setText("Blink OFF")

    def _toggle_blink_running(self):
        if bool(getattr(self, "_blink_running", False)):
            self._stop_blink()
            self._set_status("Blink OFF")
        else:
            self._start_blink()

    def _pulse_once(self):
        """A quick blink pulse after edit (non-blocking)."""
        if self.layer is None:
            return
        try:
            self.layer.visible = False
            # 120ms 后恢复
            QTimer.singleShot(
                120, lambda: setattr(self.layer, "visible", True)
            )
        except (AttributeError, RuntimeError, TypeError):
            pass

    # -------------------------
    # napari widget factory (MUST be top-level)
    # -------------------------

    # ---------- fill closed shape / area filter ----------
    def _fill_closed_shape_local_bbox(self):
        """Fill closed region (selected label) within local bbox, memory-friendly."""
        if self.layer is None:
            self._set_status("Pick a layer first.")
            return
        sel = int(getattr(self.layer, "selected_label", 0) or 0)
        if sel <= 0:
            self._set_status("Selected label must be > 0.")
            return

        arr = self.layer.data
        yy, xx = np.where(arr == sel)
        if yy.size == 0:
            self._set_status(f"Label {sel} not found.")
            return

        y0, y1 = int(yy.min()), int(yy.max())
        x0, x1 = int(xx.min()), int(xx.max())

        # 给一点边界余量，避免线贴边导致“外部/内部”判断失败
        pad = 2
        y0p = max(0, y0 - pad)
        y1p = min(arr.shape[0] - 1, y1 + pad)
        x0p = max(0, x0 - pad)
        x1p = min(arr.shape[1] - 1, x1 + pad)

        sub = np.asarray(arr[y0p : y1p + 1, x0p : x1p + 1])
        mask_line = sub == sel

        # 闭合填充思路：先找“非线区域”，从边界泛洪标记外部，剩下就是内部洞
        free = ~mask_line
        if not np.any(free):
            self._set_status("No fillable region (all selected label).")
            return

        h, w = free.shape
        seed = np.zeros_like(free, dtype=bool)
        seed[0, :] = True
        seed[-1, :] = True
        seed[:, 0] = True
        seed[:, -1] = True
        seed &= free

        outside = ndi.binary_propagation(seed, mask=free)
        inside = free & (~outside)

        # 只填“真正新增”的内部区域（避免重复写）
        to_fill = inside & (sub != sel)
        n_add = int(np.count_nonzero(to_fill))
        if n_add == 0:
            self._set_status("No enclosed region to fill (maybe not closed).")
            return

        # Undo/Redo 记录（如果你已经加了历史功能）
        try:
            yyi, xxi = np.where(to_fill)
            yy_abs = yyi + y0p
            xx_abs = xxi + x0p
            before_vals = arr[yy_abs, xx_abs].copy()
            after_vals = np.full_like(before_vals, sel)
            if hasattr(self, "_push_history"):
                self._push_history(
                    {
                        "op": f"fill closed -> {sel}",
                        "yy": yy_abs,
                        "xx": xx_abs,
                        "before_vals": before_vals,
                        "after_vals": after_vals,
                    }
                )
        except Exception:
            pass

        with self.layer.events.data.blocker():
            sub[to_fill] = sel

        self._refresh_labels_layer(pulse=True)
        self._set_status(
            f"Filled closed region for label {sel}; added_pixels={n_add}"
        )

    def _make_btn_fill_closed(self):
        @magicgui(call_button="Fill closed shape (Shift+F)")
        def _btn():
            self._fill_closed_shape_local_bbox()

        return _btn

    def _filter_small_labels_new_layer_impl(self, min_area: int = 50):
        """Create a NEW labels layer with labels smaller than min_area removed."""
        if self.layer is None:
            self._set_status("Pick a layer first.")
            return
        if int(min_area) <= 0:
            self._set_status("min_area must be > 0.")
            return

        arr = np.asarray(self.layer.data)
        if arr.ndim != 2:
            self._set_status("Only 2D labels are supported for now.")
            return

        mx = int(arr.max()) if arr.size else 0
        if mx <= 0:
            self._set_status("No labels to filter.")
            return

        # bincount 非常省内存/快速（按 label ID 统计面积）
        counts = np.bincount(arr.ravel().astype(np.int64), minlength=mx + 1)
        keep = counts >= int(min_area)
        keep[0] = True  # 背景保留

        out = arr.copy()
        out[~keep[out]] = 0

        new_name = (
            f"{getattr(self.layer, 'name', 'labels')}__minA{int(min_area)}"
        )
        L = self.viewer.add_labels(
            out.astype(np.int32, copy=False), name=new_name
        )
        try:
            L.opacity = getattr(self.layer, "opacity", 0.6)
        except Exception:
            pass

        removed_n = int(
            np.sum((counts[1:] > 0) & (counts[1:] < int(min_area)))
        )
        kept_n = int(np.sum(counts[1:] >= int(min_area)))
        self._set_status(
            f"New layer: {new_name} | min_area={int(min_area)} | kept={kept_n} | removed={removed_n}"
        )

    def _make_btn_filter_area(self):
        @magicgui(
            min_area={
                "label": "Min area",
                "min": 1,
                "max": 100000000,
                "step": 1,
            },
            call_button="Filter small labels -> NEW layer",
        )
        def _btn(min_area: int = 50):
            self._filter_small_labels_new_layer_impl(int(min_area))

        return _btn

    # ---------- keyboard shortcuts ----------
    def _bind_layer_hotkeys(self) -> None:
        # Bind hotkeys on the active Labels layer (higher priority than viewer)
        if self.layer is None:
            return
        L = self.layer

        @L.bind_key("n", overwrite=True)
        def _lk_new_id(layer):
            self._new_id()

        @L.bind_key("f", overwrite=True)
        def _lk_fill(layer):
            self._fill_closed_shape_local_bbox()

        @L.bind_key("Shift-f", overwrite=True)
        def _lk_fill_next(layer):
            self._fill_closed_shape_local_bbox()
            self._new_id()

    def _register_shortcuts(self) -> None:
        v = self.viewer

        @v.bind_key("n", overwrite=True)
        def _k_new_id(viewer):
            # N = new ID (max+1)
            self._new_id()

        @v.bind_key("f", overwrite=True)
        def _k_fill_closed(viewer):
            # F = fill closed shape (your existing implementation)
            self._fill_closed_shape_local_bbox()

        @v.bind_key("Shift-f", overwrite=True)
        def _k_fill_and_next(viewer):
            # Shift+F = fill then auto-next-id
            self._fill_closed_shape_local_bbox()
            self._new_id()


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
