from __future__ import annotations

import argparse
import os
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


UP_MODE_OPTIONS = [
    ("禁止", 0),
    ("跳跃", 1),
    ("短伸腿", 2),
    ("长伸腿", 3),
]

DOWN_MODE_OPTIONS = [
    ("禁止", 0),
    ("跳跃", 1),
    ("短伸腿", 2),
]

SPEED_OPTIONS = [
    ("1.0 m/s", 0),
    ("1.6 m/s", 1),
    ("2.2 m/s", 2),
    ("2.8 m/s", 3),
]

UP_MODE_LABEL_TO_VALUE = {label: value for label, value in UP_MODE_OPTIONS}
UP_MODE_VALUE_TO_LABEL = {value: label for label, value in UP_MODE_OPTIONS}
DOWN_MODE_LABEL_TO_VALUE = {label: value for label, value in DOWN_MODE_OPTIONS}
DOWN_MODE_VALUE_TO_LABEL = {value: label for label, value in DOWN_MODE_OPTIONS}
SPEED_LABEL_TO_VALUE = {label: value for label, value in SPEED_OPTIONS}
SPEED_VALUE_TO_LABEL = {value: label for label, value in SPEED_OPTIONS}

BLINK_INTERVAL_MS = 450
HIGHLIGHT_COLORS = ("#ffd54f", "#ff7043")
MASK_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))


@dataclass
class StepAnnotation:
    up_mode: int = 0
    down_mode: int = 0
    up_speed: int = 0
    down_speed: int = 0


@dataclass
class StepRegion:
    index: int
    ys: np.ndarray
    xs: np.ndarray
    bbox: tuple[int, int, int, int]
    area: int
    annotation: StepAnnotation
    has_mixed_alpha: bool = False
    had_reserved_bits: bool = False


def is_step_field_pixel(b_channel: np.ndarray, g_channel: np.ndarray) -> np.ndarray:
    zero_mask = (b_channel == 0) & (g_channel == 0)
    neutral_mask = (b_channel == 128) & (g_channel == 128)
    return ~(zero_mask | neutral_mask)


def encode_annotation(annotation: StepAnnotation) -> int:
    return (
        (annotation.up_mode & 0b11)
        | ((annotation.down_mode & 0b11) << 2)
        | ((annotation.up_speed & 0b11) << 4)
        | ((annotation.down_speed & 0b11) << 6)
    )


def decode_annotation(alpha_value: int) -> tuple[StepAnnotation, bool]:
    up_mode = alpha_value & 0b11
    down_mode = (alpha_value >> 2) & 0b11
    up_speed = (alpha_value >> 4) & 0b11
    down_speed = (alpha_value >> 6) & 0b11

    had_reserved_bits = down_mode == 0b11
    if down_mode == 0b11:
        down_mode = 0

    return StepAnnotation(up_mode, down_mode, up_speed, down_speed), had_reserved_bits


def dominant_annotation(alpha_values: np.ndarray) -> tuple[StepAnnotation, bool, bool]:
    unique_values = np.unique(alpha_values)
    has_mixed_alpha = unique_values.size > 1

    non_zero = alpha_values[alpha_values != 0]
    if non_zero.size == 0:
        return StepAnnotation(), has_mixed_alpha, False

    values, counts = np.unique(non_zero, return_counts=True)
    dominant_value = int(values[np.argmax(counts)])
    annotation, had_reserved_bits = decode_annotation(dominant_value)
    return annotation, has_mixed_alpha, had_reserved_bits


class StepModeAnnotatorApp:
    def __init__(self, root: tk.Tk, initial_path: str | None = None):
        self.root = root
        self.root.title("台阶模式标注器")
        self.root.geometry("1680x1020")

        self.image_path: str | None = None
        self.save_path: str | None = None
        self.bgr_image: np.ndarray | None = None
        self.loaded_alpha: np.ndarray | None = None
        self.regions: list[StepRegion] = []
        self.current_region_index = 0

        self.scale = 1.0
        self.tk_image: ImageTk.PhotoImage | None = None
        self.highlight_rect_id: int | None = None
        self.highlight_phase = False
        self.highlight_job: str | None = None
        self._suspend_ui_events = False

        self.up_mode_var = tk.StringVar(value=UP_MODE_VALUE_TO_LABEL[0])
        self.down_mode_var = tk.StringVar(value=DOWN_MODE_VALUE_TO_LABEL[0])
        self.up_speed_var = tk.StringVar(value=SPEED_VALUE_TO_LABEL[0])
        self.down_speed_var = tk.StringVar(value=SPEED_VALUE_TO_LABEL[0])
        self.jump_var = tk.StringVar(value="1")

        self._init_ui()
        self._schedule_highlight_blink()

        if initial_path:
            self.load_image(initial_path)

    def _init_ui(self):
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(toolbar, text="打开", command=self.open_image).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="另存为", command=self.save_image).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Frame(toolbar, width=12).pack(side=tk.LEFT)
        tk.Button(toolbar, text="Z+", width=4, command=self.zoom_in).pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text="Z-", width=4, command=self.zoom_out).pack(side=tk.LEFT, padx=2, pady=2)

        self.path_label = tk.Label(toolbar, text="未加载图片", anchor="w")
        self.path_label.pack(side=tk.LEFT, padx=12)

        content = tk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True)

        canvas_frame = tk.Frame(content)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.v_scroll = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        self.h_scroll = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        self.canvas = tk.Canvas(
            canvas_frame,
            bg="#2a2a2a",
            yscrollcommand=self.v_scroll.set,
            xscrollcommand=self.h_scroll.set,
        )
        self.v_scroll.config(command=self.canvas.yview)
        self.h_scroll.config(command=self.canvas.xview)

        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        panel = tk.Frame(content, width=360, padx=12, pady=12)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)

        tk.Label(panel, text="台阶区域", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")

        self.region_count_label = tk.Label(panel, text="检测到: 0")
        self.region_count_label.pack(anchor="w", pady=(8, 0))

        self.region_index_label = tk.Label(panel, text="当前: - / -")
        self.region_index_label.pack(anchor="w", pady=(4, 0))

        nav_frame = tk.Frame(panel)
        nav_frame.pack(fill=tk.X, pady=(10, 0))
        self.prev_button = tk.Button(nav_frame, text="上一块", command=self.show_previous_region, width=10)
        self.prev_button.pack(side=tk.LEFT)
        self.next_button = tk.Button(nav_frame, text="下一块", command=self.show_next_region, width=10)
        self.next_button.pack(side=tk.LEFT, padx=6)

        jump_frame = tk.Frame(panel)
        jump_frame.pack(fill=tk.X, pady=(10, 0))
        tk.Label(jump_frame, text="跳转到").pack(side=tk.LEFT)
        self.jump_entry = tk.Entry(jump_frame, textvariable=self.jump_var, width=8)
        self.jump_entry.pack(side=tk.LEFT, padx=6)
        self.jump_button = tk.Button(jump_frame, text="跳转", command=self.jump_to_region, width=8)
        self.jump_button.pack(side=tk.LEFT)

        tk.Label(panel, text="区域信息", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(16, 0))
        self.region_info_label = tk.Label(panel, text="-", justify=tk.LEFT, anchor="w")
        self.region_info_label.pack(fill=tk.X, pady=(6, 0))

        tk.Label(panel, text="标注", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(16, 0))

        self.up_mode_box = self._create_labeled_combobox(panel, "上台阶模式", self.up_mode_var, [label for label, _ in UP_MODE_OPTIONS])
        self.down_mode_box = self._create_labeled_combobox(panel, "下台阶模式", self.down_mode_var, [label for label, _ in DOWN_MODE_OPTIONS])
        self.up_speed_box = self._create_labeled_combobox(panel, "上台阶速度", self.up_speed_var, [label for label, _ in SPEED_OPTIONS])
        self.down_speed_box = self._create_labeled_combobox(panel, "下台阶速度", self.down_speed_var, [label for label, _ in SPEED_OPTIONS])

        self.up_mode_box.bind("<<ComboboxSelected>>", self.on_annotation_changed)
        self.down_mode_box.bind("<<ComboboxSelected>>", self.on_annotation_changed)
        self.up_speed_box.bind("<<ComboboxSelected>>", self.on_annotation_changed)
        self.down_speed_box.bind("<<ComboboxSelected>>", self.on_annotation_changed)

        self.annotation_label = tk.Label(panel, text="Alpha 编码: 0x00 (未设置)", anchor="w")
        self.annotation_label.pack(fill=tk.X, pady=(10, 0))

        batch_frame = tk.Frame(panel)
        batch_frame.pack(fill=tk.X, pady=(14, 0))
        self.apply_all_button = tk.Button(
            batch_frame,
            text="当前配置应用到全部",
            command=self.apply_current_annotation_to_all,
            width=22,
        )
        self.apply_all_button.pack(anchor="w")

        self.status_label = tk.Label(panel, text="加载 PNG 后开始标注", justify=tk.LEFT, anchor="w", fg="#1f5fa7")
        self.status_label.pack(fill=tk.X, pady=(18, 0))

        status_bar = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.pointer_label = tk.Label(status_bar, text="X:-, Y:-", anchor="w", width=18)
        self.pointer_label.pack(side=tk.LEFT, padx=6)
        self.pixel_label = tk.Label(status_bar, text="B:-, G:-, R:-, A:-", anchor="w")
        self.pixel_label.pack(side=tk.LEFT, padx=6)

        self.canvas.bind("<Motion>", self.on_canvas_mouse_move)
        self.canvas.bind("<Leave>", self.on_canvas_mouse_leave)
        self.jump_entry.bind("<Return>", lambda _event: self.jump_to_region())

        self._set_controls_enabled(False)

    def _create_labeled_combobox(self, parent, label_text, variable, values):
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, pady=(8, 0))
        tk.Label(frame, text=label_text, width=12, anchor="w").pack(side=tk.LEFT)
        combo = ttk.Combobox(frame, textvariable=variable, values=values, state="readonly", width=18)
        combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return combo

    def _set_controls_enabled(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        combo_state = "readonly" if enabled else tk.DISABLED
        for button in (self.prev_button, self.next_button, self.jump_button, self.apply_all_button):
            button.config(state=state)
        self.jump_entry.config(state=state)
        self.up_mode_box.config(state=combo_state)
        self.down_mode_box.config(state=combo_state)
        self.up_speed_box.config(state=combo_state)
        self.down_speed_box.config(state=combo_state)

    def _schedule_highlight_blink(self):
        if self.bgr_image is not None:
            self.highlight_phase = not self.highlight_phase
            self._update_highlight_visual()
        self.highlight_job = self.root.after(BLINK_INTERVAL_MS, self._schedule_highlight_blink)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("PNG", "*.png")])
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path: str):
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            messagebox.showerror("打开失败", f"无法打开图片:\n{file_path}")
            return

        if image.ndim != 3 or image.shape[2] not in (3, 4):
            messagebox.showerror(
                "打开失败",
                "仅支持 3 通道或 4 通道 PNG 图片。",
            )
            return

        if image.shape[2] == 3:
            bgr_image = image.copy()
            alpha = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            bgr_image = image[:, :, :3].copy()
            alpha = image[:, :, 3].copy()

        self.image_path = file_path
        self.save_path = file_path
        self.bgr_image = bgr_image
        self.loaded_alpha = alpha
        self.scale = 1.0
        self.current_region_index = 0
        self.regions = self._extract_regions(alpha)

        self.path_label.config(text=file_path)
        self.region_count_label.config(text=f"检测到: {len(self.regions)}")

        self.refresh_canvas()

        if self.regions:
            self._set_controls_enabled(True)
            self.show_region(0)
            self.status_label.config(
                text="已根据 BG 方向场检测台阶区域。\n原始台阶像素之外的 alpha 会保持为 0。",
                fg="#1f5fa7",
            )
        else:
            self._set_controls_enabled(False)
            self._clear_region_panel()
            self.status_label.config(
                text="未检测到台阶区域。\n保存时仍会导出 alpha 全 0 的 RGBA 图。",
                fg="#8a5d00",
            )

    def _extract_regions(self, alpha: np.ndarray) -> list[StepRegion]:
        assert self.bgr_image is not None

        b_channel = self.bgr_image[:, :, 0]
        g_channel = self.bgr_image[:, :, 1]
        raw_mask = is_step_field_pixel(b_channel, g_channel)
        closed_mask = cv2.morphologyEx(raw_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, MASK_KERNEL) > 0

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask.astype(np.uint8), connectivity=8)

        regions: list[StepRegion] = []
        for label in range(1, component_count):
            component_mask = labels == label
            region_mask = component_mask & raw_mask
            if not np.any(region_mask):
                continue

            ys, xs = np.where(region_mask)
            annotation, has_mixed_alpha, had_reserved_bits = dominant_annotation(alpha[ys, xs])

            left = int(stats[label, cv2.CC_STAT_LEFT])
            top = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])

            regions.append(
                StepRegion(
                    index=len(regions),
                    ys=ys,
                    xs=xs,
                    bbox=(left, top, width, height),
                    area=int(ys.size),
                    annotation=annotation,
                    has_mixed_alpha=has_mixed_alpha,
                    had_reserved_bits=had_reserved_bits,
                )
            )

        regions.sort(key=lambda region: (region.bbox[1], region.bbox[0]))
        for index, region in enumerate(regions):
            region.index = index
        return regions

    def save_image(self):
        if self.bgr_image is None:
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            initialfile=self._default_save_name(),
        )
        if not file_path:
            return

        height, width = self.bgr_image.shape[:2]
        output_alpha = np.zeros((height, width), dtype=np.uint8)
        invalid_regions = []

        for region in self.regions:
            encoded = encode_annotation(region.annotation)
            if encoded == 0:
                invalid_regions.append(region.index + 1)
            output_alpha[region.ys, region.xs] = encoded

        if invalid_regions:
            preview = ", ".join(str(idx) for idx in invalid_regions[:12])
            suffix = "" if len(invalid_regions) <= 12 else ", ..."
            messagebox.showerror(
                "保存被阻止",
                (
                    "台阶区域不能以 alpha = 0 保存。\n"
                    "请先标注这些区域: "
                    f"{preview}{suffix}"
                ),
            )
            return

        output = np.dstack((self.bgr_image, output_alpha))
        if not cv2.imwrite(file_path, output):
            messagebox.showerror("保存失败", f"无法保存图片:\n{file_path}")
            return

        self.save_path = file_path
        self.status_label.config(text=f"已保存 RGBA 地图到 {file_path}", fg="#1a7f37")
        messagebox.showinfo("保存成功", f"已保存 RGBA 地图:\n{file_path}")

    def _default_save_name(self) -> str:
        if not self.image_path:
            return "annotated_step_map.png"
        base_name = os.path.basename(self.image_path)
        stem, _ = os.path.splitext(base_name)
        return f"{stem}_step_modes.png"

    def zoom_in(self):
        if self.bgr_image is None:
            return
        self.scale *= 1.5
        self.refresh_canvas()

    def zoom_out(self):
        if self.bgr_image is None:
            return
        self.scale = max(self.scale / 1.5, 0.25)
        self.refresh_canvas()

    def refresh_canvas(self):
        if self.bgr_image is None:
            self.canvas.delete("all")
            return

        rgb_image = cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2RGB)
        rgb_image = np.flipud(rgb_image)

        height, width = rgb_image.shape[:2]
        scaled_width = max(1, int(width * self.scale))
        scaled_height = max(1, int(height * self.scale))

        display_image = Image.fromarray(rgb_image).resize((scaled_width, scaled_height), Image.NEAREST)
        self.tk_image = ImageTk.PhotoImage(display_image)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)
        self.canvas.config(scrollregion=(0, 0, scaled_width, scaled_height))
        self.highlight_rect_id = None
        self._draw_or_update_highlight()

    def image_to_display_row(self, y_img: int) -> int:
        assert self.bgr_image is not None
        height = self.bgr_image.shape[0]
        return (height - 1) - y_img

    def canvas_to_image_xy(self, event) -> tuple[int, int] | None:
        if self.bgr_image is None:
            return None

        height, width = self.bgr_image.shape[:2]
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        x_img = int(canvas_x / self.scale)
        y_disp = int(canvas_y / self.scale)
        y_img = self.image_to_display_row(y_disp)

        if x_img < 0 or y_img < 0 or x_img >= width or y_img >= height:
            return None
        return x_img, y_img

    def on_canvas_mouse_move(self, event):
        image_xy = self.canvas_to_image_xy(event)
        if image_xy is None or self.bgr_image is None or self.loaded_alpha is None:
            self.pointer_label.config(text="X:-, Y:-")
            self.pixel_label.config(text="B:-, G:-, R:-, A:-")
            return

        x_img, y_img = image_xy
        b_value, g_value, r_value = [int(value) for value in self.bgr_image[y_img, x_img]]
        a_value = int(self.loaded_alpha[y_img, x_img])
        self.pointer_label.config(text=f"X:{x_img}, Y:{y_img}")
        self.pixel_label.config(text=f"B:{b_value}, G:{g_value}, R:{r_value}, A:{a_value}")

    def on_canvas_mouse_leave(self, _event):
        self.pointer_label.config(text="X:-, Y:-")
        self.pixel_label.config(text="B:-, G:-, R:-, A:-")

    def show_previous_region(self):
        if self.regions:
            self.show_region((self.current_region_index - 1) % len(self.regions))

    def show_next_region(self):
        if self.regions:
            self.show_region((self.current_region_index + 1) % len(self.regions))

    def jump_to_region(self):
        if not self.regions:
            return

        try:
            target_index = int(self.jump_var.get()) - 1
        except ValueError:
            messagebox.showerror("区域索引无效", "请输入有效的区域编号。")
            return

        if target_index < 0 or target_index >= len(self.regions):
            messagebox.showerror("区域索引无效", f"区域编号必须在 [1, {len(self.regions)}] 范围内。")
            return

        self.show_region(target_index)

    def show_region(self, region_index: int):
        if not self.regions:
            self.current_region_index = 0
            self._clear_region_panel()
            self._draw_or_update_highlight()
            return

        self.current_region_index = region_index
        region = self.regions[region_index]

        self.region_index_label.config(text=f"当前: {region_index + 1} / {len(self.regions)}")
        self.jump_var.set(str(region_index + 1))

        x, y, width, height = region.bbox
        info_lines = [
            f"包围框: x={x}, y={y}, w={width}, h={height}",
            f"原始像素数: {region.area}",
        ]
        if region.has_mixed_alpha:
            info_lines.append("已有 alpha 不一致，已回填为占比最高的非零值")
        if region.had_reserved_bits:
            info_lines.append("下台阶检测到保留模式位(3)，已按禁止处理")
        if encode_annotation(region.annotation) == 0:
            info_lines.append("状态: 未设置（保存会被阻止）")
        self.region_info_label.config(text="\n".join(info_lines))

        self._load_region_annotation_into_ui(region.annotation)
        self._update_annotation_label(region.annotation)
        self._draw_or_update_highlight()
        self._scroll_current_region_into_view()

    def _clear_region_panel(self):
        self.region_index_label.config(text="当前: - / -")
        self.region_info_label.config(text="-")
        self._load_region_annotation_into_ui(StepAnnotation())
        self.annotation_label.config(text="Alpha 编码: 0x00 (未设置)")

    def _load_region_annotation_into_ui(self, annotation: StepAnnotation):
        self._suspend_ui_events = True
        try:
            self.up_mode_var.set(UP_MODE_VALUE_TO_LABEL.get(annotation.up_mode, UP_MODE_VALUE_TO_LABEL[0]))
            self.down_mode_var.set(DOWN_MODE_VALUE_TO_LABEL.get(annotation.down_mode, DOWN_MODE_VALUE_TO_LABEL[0]))
            self.up_speed_var.set(SPEED_VALUE_TO_LABEL.get(annotation.up_speed, SPEED_VALUE_TO_LABEL[0]))
            self.down_speed_var.set(SPEED_VALUE_TO_LABEL.get(annotation.down_speed, SPEED_VALUE_TO_LABEL[0]))
        finally:
            self._suspend_ui_events = False

    def _sync_loaded_alpha(self):
        if self.loaded_alpha is None:
            return
        self.loaded_alpha.fill(0)
        for region in self.regions:
            encoded = encode_annotation(region.annotation)
            if encoded:
                self.loaded_alpha[region.ys, region.xs] = encoded

    def on_annotation_changed(self, _event=None):
        if self._suspend_ui_events or not self.regions:
            return

        annotation = self._annotation_from_ui()
        self.regions[self.current_region_index].annotation = annotation
        self._sync_loaded_alpha()
        self._update_annotation_label(annotation)

        if encode_annotation(annotation) == 0:
            self.status_label.config(
                text=(
                    f"区域 {self.current_region_index + 1} 仍未设置。"
                    "alpha 变为非零前无法保存。"
                ),
                fg="#8a5d00",
            )
        else:
            self.status_label.config(
                text=f"已更新区域 {self.current_region_index + 1} 的标注。",
                fg="#1f5fa7",
            )
        self.show_region(self.current_region_index)

    def _annotation_from_ui(self) -> StepAnnotation:
        return StepAnnotation(
            up_mode=UP_MODE_LABEL_TO_VALUE[self.up_mode_var.get()],
            down_mode=DOWN_MODE_LABEL_TO_VALUE[self.down_mode_var.get()],
            up_speed=SPEED_LABEL_TO_VALUE[self.up_speed_var.get()],
            down_speed=SPEED_LABEL_TO_VALUE[self.down_speed_var.get()],
        )

    def _update_annotation_label(self, annotation: StepAnnotation):
        encoded = encode_annotation(annotation)
        if encoded == 0:
            self.annotation_label.config(text="Alpha 编码: 0x00 (未设置)")
            return
        bits = format(encoded, "08b")
        self.annotation_label.config(text=f"Alpha 编码: 0x{encoded:02X} ({encoded}) bits={bits}")

    def apply_current_annotation_to_all(self):
        if not self.regions:
            return

        annotation = self._annotation_from_ui()
        if not messagebox.askyesno(
            "批量设置",
            "将当前标注应用到所有检测到的区域？",
        ):
            return

        for region in self.regions:
            region.annotation = StepAnnotation(
                up_mode=annotation.up_mode,
                down_mode=annotation.down_mode,
                up_speed=annotation.up_speed,
                down_speed=annotation.down_speed,
            )

        self._sync_loaded_alpha()
        self.status_label.config(text="已将当前标注应用到全部区域。", fg="#1a7f37")
        self.show_region(self.current_region_index)

    def _draw_or_update_highlight(self):
        if self.bgr_image is None or not self.regions:
            if self.highlight_rect_id is not None:
                self.canvas.delete(self.highlight_rect_id)
                self.highlight_rect_id = None
            return

        region = self.regions[self.current_region_index]
        x, y, width, height = region.bbox
        left = x * self.scale
        right = (x + width) * self.scale
        top = self.image_to_display_row(y + height - 1) * self.scale
        bottom = (self.image_to_display_row(y) + 1) * self.scale

        if self.highlight_rect_id is None:
            self.highlight_rect_id = self.canvas.create_rectangle(left, top, right, bottom)
        else:
            self.canvas.coords(self.highlight_rect_id, left, top, right, bottom)

        self._update_highlight_visual()

    def _update_highlight_visual(self):
        if self.highlight_rect_id is None:
            return

        color = HIGHLIGHT_COLORS[1 if self.highlight_phase else 0]
        width = max(2, int(round(self.scale)))
        self.canvas.itemconfig(
            self.highlight_rect_id,
            outline=color,
            fill=color,
            width=width,
            stipple="gray50",
        )

    def _scroll_current_region_into_view(self):
        if self.bgr_image is None or not self.regions:
            return

        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        image_height, image_width = self.bgr_image.shape[:2]

        x, y, width, height = self.regions[self.current_region_index].bbox
        center_x = (x + width / 2.0) * self.scale
        display_top = self.image_to_display_row(y + height - 1)
        display_bottom = self.image_to_display_row(y)
        center_y = ((display_top + display_bottom + 1) / 2.0) * self.scale

        total_width = max(image_width * self.scale, 1.0)
        total_height = max(image_height * self.scale, 1.0)

        x_fraction = (center_x - canvas_width / 2.0) / total_width
        y_fraction = (center_y - canvas_height / 2.0) / total_height
        x_fraction = min(max(x_fraction, 0.0), 1.0)
        y_fraction = min(max(y_fraction, 0.0), 1.0)
        self.canvas.xview_moveto(x_fraction)
        self.canvas.yview_moveto(y_fraction)


def parse_args():
    parser = argparse.ArgumentParser(description="在 PNG alpha 通道中标注台阶模式。")
    parser.add_argument("image", nargs="?", help="可选：3 通道或 4 通道 PNG 路径")
    return parser.parse_args()


def main():
    args = parse_args()
    root = tk.Tk()
    StepModeAnnotatorApp(root, initial_path=args.image)
    root.mainloop()


if __name__ == "__main__":
    main()
