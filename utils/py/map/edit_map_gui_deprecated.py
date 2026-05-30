import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
from PIL import Image, ImageTk, ImageDraw
import os
import math  # 新增：用于向量计算

class MapEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cost Map Editor")
        self.root.geometry("1500x1000") # 稍微加宽一点以容纳新按钮

        # State variables
        self.image_path = None
        self.original_image = None  # PIL Image
        self.display_image = None   # PIL Image for display (scaled)
        self.tk_image = None        # ImageTk for canvas
        self.draw_color = (255, 0, 0) # Default red (Pixel/Rect mode default)
        
        # Mode variables
        self.mode = "pixel"         # "pixel", "rect", "line"
        self.line_width = 1         # 直线宽度
        self.line_color_mode = "vector"  # "vector" (dir-based) | "fixed" (use selected color)
        
        self.scale = 1.0
        self.history = []           # Undo history
        
        # Mouse drag variables
        self.start_x = None
        self.start_y = None
        self.temp_draw_id = None    # 用于存储临时画的矩形或直线的ID

        # Status bar widgets
        self.status_coord_label = None
        self.status_color_label = None

        self._init_ui()

    # ---- Coordinate transforms (image <-> displayed canvas) ----
    # Original image coords: origin top-left, +x right, +y down.
    # Display: we show vertically flipped for intuitive view.
    # Mouse/status: origin bottom-left, +x right, +y up (visually),
    # but the numeric coordinate shown equals the original image (x, y)
    # after applying the display flip mapping.
    def _img_size(self):
        if self.original_image is None:
            return 0, 0
        return self.original_image.size

    def image_to_display_row(self, y_img: int) -> int:
        """Map image y (top-left origin) to displayed image row (top-left origin)."""
        _, h = self._img_size()
        return (h - 1) - y_img

    def display_row_to_image(self, y_disp_row: int) -> int:
        """Inverse of image_to_display_row."""
        _, h = self._img_size()
        return (h - 1) - y_disp_row

    def image_to_canvas_xy(self, x_img: int, y_img: int, *, center: bool = False):
        """Convert image pixel coords to canvas coords (in screen pixels)."""
        y_disp_row = self.image_to_display_row(y_img)
        off = 0.5 if center else 0.0
        return (x_img + off) * self.scale, (y_disp_row + off) * self.scale

    def canvas_to_image_xy(self, event):
        """Convert a canvas mouse event to image pixel coords."""
        if self.original_image is None:
            return None
        w, h = self._img_size()
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        x_disp = int(cx / self.scale)
        y_disp_row = int(cy / self.scale)
        x_img = x_disp
        y_img = self.display_row_to_image(y_disp_row)

        if x_img < 0 or y_img < 0 or x_img >= w or y_img >= h:
            return None
        return x_img, y_img

    def canvas_to_display_xy(self, event):
        """User-facing coords: origin bottom-left, +y up (matches what they see)."""
        if self.original_image is None:
            return None
        w, h = self._img_size()
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        x_disp = int(cx / self.scale)
        y_disp_row = int(cy / self.scale)
        x_u = x_disp
        y_u = (h - 1) - y_disp_row
        if x_u < 0 or y_u < 0 or x_u >= w or y_u >= h:
            return None
        return x_u, y_u

    def _init_ui(self):
        # --- Toolbar ---
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        # File Operations
        btn_open = tk.Button(toolbar, text="Open", command=self.open_image)
        btn_open.pack(side=tk.LEFT, padx=2, pady=2)

        btn_save = tk.Button(toolbar, text="Save", command=self.save_image)
        btn_save.pack(side=tk.LEFT, padx=2, pady=2)

        tk.Frame(toolbar, width=10).pack(side=tk.LEFT) # Spacer

        btn_undo = tk.Button(toolbar, text="Undo", command=self.undo)
        btn_undo.pack(side=tk.LEFT, padx=2, pady=2)

        tk.Frame(toolbar, width=10).pack(side=tk.LEFT) # Spacer

        # Zoom
        btn_zoom_in = tk.Button(toolbar, text="Z+", width=3, command=self.zoom_in)
        btn_zoom_in.pack(side=tk.LEFT, padx=1, pady=2)

        btn_zoom_out = tk.Button(toolbar, text="Z-", width=3, command=self.zoom_out)
        btn_zoom_out.pack(side=tk.LEFT, padx=1, pady=2)

        tk.Frame(toolbar, width=15).pack(side=tk.LEFT) # Spacer

        # --- Modes ---
        self.mode_var = tk.StringVar(value="pixel")
        
        # Mode: Pixel
        rb_pixel = tk.Radiobutton(toolbar, text="Pixel", variable=self.mode_var, 
                                  value="pixel", command=self.set_mode)
        rb_pixel.pack(side=tk.LEFT, padx=2)
        
        # Mode: Rect
        rb_rect = tk.Radiobutton(toolbar, text="Rect", variable=self.mode_var, 
                                 value="rect", command=self.set_mode)
        rb_rect.pack(side=tk.LEFT, padx=2)

        # Mode: Line (New Function)
        rb_line = tk.Radiobutton(toolbar, text="Line", variable=self.mode_var, 
                                 value="line", command=self.set_mode)
        rb_line.pack(side=tk.LEFT, padx=2)

        tk.Frame(toolbar, width=10).pack(side=tk.LEFT) # Spacer

        # --- Line Settings (Width & Direction) ---
        lbl_width = tk.Label(toolbar, text="Width:")
        lbl_width.pack(side=tk.LEFT, padx=1)
        
        self.spin_width = tk.Spinbox(toolbar, from_=1, to=50, width=3, command=self.update_line_width)
        self.spin_width.pack(side=tk.LEFT, padx=2)
        self.spin_width.delete(0, "end")
        self.spin_width.insert(0, 1)

        # Line color mode (direction-based vs fixed color)
        self.line_color_mode_var = tk.StringVar(value="vector")

        self.rb_line_vector = tk.Radiobutton(
            toolbar,
            text="Dir Color",
            variable=self.line_color_mode_var,
            value="vector",
            command=self.update_line_color_mode,
        )
        self.rb_line_vector.pack(side=tk.LEFT, padx=4)

        self.rb_line_fixed = tk.Radiobutton(
            toolbar,
            text="Fixed Color",
            variable=self.line_color_mode_var,
            value="fixed",
            command=self.update_line_color_mode,
        )
        self.rb_line_fixed.pack(side=tk.LEFT, padx=2)

        # Direction Invert Checkbox
        self.invert_dir_var = tk.BooleanVar(value=False)
        self.chk_invert = tk.Checkbutton(toolbar, text="Invert Dir", variable=self.invert_dir_var)
        self.chk_invert.pack(side=tk.LEFT, padx=5)

        tk.Frame(toolbar, width=10).pack(side=tk.LEFT) # Spacer

        # --- Color Picker (Only for Pixel/Rect mode) ---
        self.color_btn = tk.Button(toolbar, text="Color", bg="#FF0000", fg="white", width=5,
                                   command=self.choose_color)
        self.color_btn.pack(side=tk.LEFT, padx=5)

        # --- Canvas Area ---
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.v_scroll = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.h_scroll = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray",
                                yscrollcommand=self.v_scroll.set,
                                xscrollcommand=self.h_scroll.set)
        
        self.v_scroll.config(command=self.canvas.yview)
        self.h_scroll.config(command=self.canvas.xview)

        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Status Bar ---
        status_bar = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_coord_label = tk.Label(status_bar, text="X:-, Y:-", anchor="w", width=15)
        self.status_coord_label.pack(side=tk.LEFT, padx=5)

        self.status_color_label = tk.Label(status_bar, text="B:-, G:-, R:-", anchor="w")
        self.status_color_label.pack(side=tk.LEFT, padx=15)
        
        self.status_info_label = tk.Label(status_bar, text="Ready", anchor="w", fg="blue")
        self.status_info_label.pack(side=tk.RIGHT, padx=15)

        # Event Bindings
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", self.on_mouse_leave)
        
        self.root.bind("<Control-z>", lambda event: self.undo())

        # Initialize control states for the default mode (pixel)
        self.set_mode()

    def set_mode(self):
        self.mode = self.mode_var.get()
        if self.mode == "line":
            self._set_line_controls_state(tk.NORMAL)
            self.update_line_color_mode()
        else:
            self._set_line_controls_state(tk.DISABLED)
            self.status_info_label.config(text=f"Mode: {self.mode.capitalize()}")
            self.color_btn.config(state=tk.NORMAL)

    def _set_line_controls_state(self, state):
        # Widgets that only apply in line mode
        try:
            self.spin_width.config(state=state)
        except Exception:
            pass
        try:
            self.rb_line_vector.config(state=state)
            self.rb_line_fixed.config(state=state)
        except Exception:
            pass
        try:
            self.chk_invert.config(state=state)
        except Exception:
            pass

    def update_line_color_mode(self):
        # Only meaningful in line mode
        self.line_color_mode = self.line_color_mode_var.get()
        if self.mode != "line":
            return

        if self.line_color_mode == "fixed":
            self.color_btn.config(state=tk.NORMAL)
            self.chk_invert.config(state=tk.DISABLED)
            self.status_info_label.config(text="Mode: Line (Fixed Color)")
        else:
            self.color_btn.config(state=tk.DISABLED)
            self.chk_invert.config(state=tk.NORMAL)
            self.status_info_label.config(text="Mode: Line (Dir Color)")

    def update_line_width(self):
        try:
            w = int(self.spin_width.get())
            self.line_width = max(1, w)
        except ValueError:
            self.line_width = 1

    def zoom_in(self):
        self.scale *= 1.5
        self.refresh_canvas()

    def zoom_out(self):
        self.scale /= 1.5
        self.refresh_canvas()

    def undo(self):
        if not self.history:
            return
        self.original_image = self.history.pop()
        self.refresh_canvas()

    def save_state(self):
        if self.original_image:
            self.history.append(self.original_image.copy())
            if len(self.history) > 20:
                self.history.pop(0)

    def choose_color(self):
        color = colorchooser.askcolor(title="Choose Drawing Color", color=self.rgb_to_hex(self.draw_color))
        if color[1]:
            self.draw_color = tuple(map(int, color[0]))
            self.color_btn.config(bg=color[1])

    def rgb_to_hex(self, rgb):
        return "#%02x%02x%02x" % rgb

    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )
        if not file_path:
            return

        try:
            self.image_path = file_path
            self.original_image = Image.open(file_path).convert("RGB")
            self.history = []
            self.scale = 1.0
            self.refresh_canvas()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image: {e}")

    def save_image(self):
        if self.original_image is None:
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        try:
            self.original_image.save(file_path)
            messagebox.showinfo("Success", "Image saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {e}")

    def refresh_canvas(self):
        if self.original_image is None:
            return

        w, h = self.original_image.size
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)
        
        # Display is only for visualization: vertically flip for intuitive coordinate view.
        disp = self.original_image.transpose(Image.FLIP_TOP_BOTTOM)
        self.display_image = disp.resize((new_w, new_h), Image.NEAREST)
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def get_image_coords(self, event):
        # Backward-compatible name: now returns original image coords.
        xy = self.canvas_to_image_xy(event)
        if xy is None:
            return -1, -1
        return xy

    def on_mouse_move(self, event):
        if self.original_image is None:
            return

        # User-facing status coords: origin bottom-left, y up.
        disp_xy = self.canvas_to_display_xy(event)
        if disp_xy is None:
            if self.status_coord_label:
                self.status_coord_label.config(text="X:-, Y:-")
            if self.status_color_label:
                self.status_color_label.config(text="B:-, G:-, R:-")
            return

        x_u, y_u = disp_xy
        # Underlying image coords for reading pixel are the same (x, y_u)
        x, y = x_u, y_u
        w, h = self._img_size()

        if self.status_coord_label:
            self.status_coord_label.config(text=f"X:{x_u}, Y:{y_u}")

        if 0 <= x < w and 0 <= y < h:
            try:
                r, g, b = self.original_image.getpixel((x, y))
                if self.status_color_label:
                    # 显示顺序按照用户习惯 B, G, R
                    self.status_color_label.config(text=f"B:{b}, G:{g}, R:{r}")
            except Exception:
                pass
        else:
            if self.status_color_label:
                self.status_color_label.config(text="B:-, G:-, R:-")

    def on_mouse_leave(self, event):
        if self.status_coord_label:
            self.status_coord_label.config(text="X:-, Y:-")
        if self.status_color_label:
            self.status_color_label.config(text="B:-, G:-, R:-")

    def on_mouse_down(self, event):
        if self.original_image is None:
            return

        xy = self.canvas_to_image_xy(event)
        if xy is None:
            return

        self.save_state()

        x, y = xy
        w, h = self._img_size()
        
        if self.mode == "pixel":
            self.paint_pixel(x, y)
        elif self.mode == "rect" or self.mode == "line":
            self.start_x = x
            self.start_y = y
            # Initialize temp drawing
            sx, sy = self.image_to_canvas_xy(x, y, center=False)
            if self.mode == "rect":
                self.temp_draw_id = self.canvas.create_rectangle(sx, sy, sx, sy, outline="blue", width=2)
            elif self.mode == "line":
                # Line drawing preview
                cx, cy = self.image_to_canvas_xy(x, y, center=True)
                preview_color = "cyan"
                if self.line_color_mode == "fixed":
                    preview_color = self.rgb_to_hex(self.draw_color)
                self.temp_draw_id = self.canvas.create_line(
                    cx,
                    cy,
                    cx,
                    cy,
                    fill=preview_color,
                    width=max(1, self.line_width * self.scale),
                )

    def on_mouse_drag(self, event):
        if self.original_image is None:
            return

        xy = self.canvas_to_image_xy(event)
        if xy is None:
            # Clamp to nearest edge for smoother preview when dragging out of bounds
            w, h = self._img_size()
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            x_disp = int(cx / self.scale)
            y_disp_row = int(cy / self.scale)
            x_disp = max(0, min(w - 1, x_disp))
            y_disp_row = max(0, min(h - 1, y_disp_row))
            x = x_disp
            y = self.display_row_to_image(y_disp_row)
        else:
            x, y = xy

        if self.mode == "pixel":
            self.paint_pixel(x, y)
        
        elif self.mode == "rect" and self.temp_draw_id:
            x0, x1 = min(self.start_x, x), max(self.start_x, x)
            y0, y1 = min(self.start_y, y), max(self.start_y, y)

            # Pixel-edge aligned rectangle: [x0, y0]..[x1, y1] inclusive
            left = x0 * self.scale
            right = (x1 + 1) * self.scale
            top = self.image_to_display_row(y1) * self.scale
            bottom = (self.image_to_display_row(y0) + 1) * self.scale
            self.canvas.coords(self.temp_draw_id, left, top, right, bottom)
            
        elif self.mode == "line" and self.temp_draw_id:
            # Update line preview coordinates
            sx, sy = self.image_to_canvas_xy(self.start_x, self.start_y, center=True)
            ex, ey = self.image_to_canvas_xy(x, y, center=True)
            self.canvas.coords(self.temp_draw_id, sx, sy, ex, ey)

    def on_mouse_up(self, event):
        if self.original_image is None:
            return

        xy = self.canvas_to_image_xy(event)
        if xy is None:
            w, h = self._img_size()
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            x_disp = int(cx / self.scale)
            y_disp_row = int(cy / self.scale)
            x_disp = max(0, min(w - 1, x_disp))
            y_disp_row = max(0, min(h - 1, y_disp_row))
            x = x_disp
            y = self.display_row_to_image(y_disp_row)
        else:
            x, y = xy

        if self.mode == "rect" and self.temp_draw_id:
            draw = ImageDraw.Draw(self.original_image)
            x0, y0 = min(self.start_x, x), min(self.start_y, y)
            x1, y1 = max(self.start_x, x), max(self.start_y, y)
            draw.rectangle([x0, y0, x1, y1], fill=self.draw_color, outline=None)
            
            self.canvas.delete(self.temp_draw_id)
            self.temp_draw_id = None
            self.refresh_canvas()
            
        elif self.mode == "line" and self.temp_draw_id:
            if self.line_color_mode == "fixed":
                line_color = tuple(self.draw_color)
            else:
                # 1. 计算向量
                dx = x - self.start_x
                dy = y - self.start_y

                # 2. 处理反向
                if self.invert_dir_var.get():
                    dx = -dx
                    dy = -dy

                # 3. 计算颜色
                length = math.sqrt(dx * dx + dy * dy)

                # 默认中性色 (当长度为0时，也就是只点了一下没有拖拽)
                line_color = (0, 128, 128)  # R=0, G=128, B=128

                if length > 0:
                    # 归一化并向左旋转90度
                    nx = -dy / length
                    ny = dx / length

                    # 映射到 [1, 255]，中心 128
                    # 需求: B对应X正方向(255)，G对应Y方向(Y正为下，图像坐标Y向下为正)
                    # 映射公式: val = 128 + dir * 127
                    val_x = int(128 + nx * 127)
                    val_y = int(128 + ny * 127)

                    # 钳制范围
                    val_x = max(1, min(255, val_x))
                    val_y = max(1, min(255, val_y))

                    # 组合颜色 (R, G, B) - Blue=X, Green=Y, Red=0
                    line_color = (0, val_y, val_x)

            # 4. 在原图上画线
            draw = ImageDraw.Draw(self.original_image)
            draw.line([(self.start_x, self.start_y), (x, y)], fill=line_color, width=self.line_width)

            # 5. 清理 UI
            self.canvas.delete(self.temp_draw_id)
            self.temp_draw_id = None

            # 6. 更新状态栏信息告诉用户刚才画的颜色
            self.status_info_label.config(
                text=f"Line Drawn. Color: B={line_color[2]}, G={line_color[1]}, R={line_color[0]}"
            )

            self.refresh_canvas()

    def paint_pixel(self, x, y):
        try:
            self.original_image.putpixel((x, y), self.draw_color)
            sx, sy = self.image_to_canvas_xy(x, y, center=False)
            self.canvas.create_rectangle(sx, sy, sx + self.scale, sy + self.scale, 
                                         outline="", fill=self.rgb_to_hex(self.draw_color))
        except IndexError:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = MapEditorApp(root)
    root.mainloop()