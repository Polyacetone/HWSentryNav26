import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
from PIL import Image, ImageTk, ImageDraw
import os

class MapEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cost Map Editor")
        self.root.geometry("1000x700")

        # State variables
        self.image_path = None
        self.original_image = None  # PIL Image
        self.display_image = None   # PIL Image for display (scaled)
        self.tk_image = None        # ImageTk for canvas
        self.draw_color = (255, 0, 0) # Default red
        self.mode = "pixel"         # "pixel" or "rect"
        self.scale = 1.0
        self.history = []           # Undo history
        
        # Mouse drag variables
        self.start_x = None
        self.start_y = None
        self.rect_id = None

        # Status bar widgets
        self.status_coord_label = None
        self.status_color_label = None

        self._init_ui()

    def _init_ui(self):
        # --- Toolbar ---
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        btn_open = tk.Button(toolbar, text="Open Image", command=self.open_image)
        btn_open.pack(side=tk.LEFT, padx=2, pady=2)

        btn_save = tk.Button(toolbar, text="Save Image", command=self.save_image)
        btn_save.pack(side=tk.LEFT, padx=2, pady=2)

        tk.Frame(toolbar, width=10).pack(side=tk.LEFT) # Spacer

        btn_undo = tk.Button(toolbar, text="Undo", command=self.undo)
        btn_undo.pack(side=tk.LEFT, padx=2, pady=2)

        tk.Frame(toolbar, width=10).pack(side=tk.LEFT) # Spacer

        btn_zoom_in = tk.Button(toolbar, text="Zoom In (+)", command=self.zoom_in)
        btn_zoom_in.pack(side=tk.LEFT, padx=2, pady=2)

        btn_zoom_out = tk.Button(toolbar, text="Zoom Out (-)", command=self.zoom_out)
        btn_zoom_out.pack(side=tk.LEFT, padx=2, pady=2)

        tk.Frame(toolbar, width=20).pack(side=tk.LEFT) # Spacer

        self.mode_var = tk.StringVar(value="pixel")
        rb_pixel = tk.Radiobutton(toolbar, text="Pixel Mode", variable=self.mode_var, 
                                  value="pixel", command=self.set_mode)
        rb_pixel.pack(side=tk.LEFT, padx=5)
        
        rb_rect = tk.Radiobutton(toolbar, text="Rectangle Mode", variable=self.mode_var, 
                                 value="rect", command=self.set_mode)
        rb_rect.pack(side=tk.LEFT, padx=5)

        tk.Frame(toolbar, width=20).pack(side=tk.LEFT) # Spacer

        self.color_btn = tk.Button(toolbar, text="Select Color", bg="#FF0000", fg="white", 
                                   command=self.choose_color)
        self.color_btn.pack(side=tk.LEFT, padx=5)

        # --- Canvas Area ---
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Scrollbars
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

        self.status_coord_label = tk.Label(status_bar, text="X:-, Y:-", anchor="w")
        self.status_coord_label.pack(side=tk.LEFT, padx=5)

        self.status_color_label = tk.Label(status_bar, text="B:-, G:-, R:-", anchor="w")
        self.status_color_label.pack(side=tk.LEFT, padx=15)

        # Event Bindings
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        # 鼠标移动时更新状态栏
        self.canvas.bind("<Motion>", self.on_mouse_move)
        # 鼠标离开画布时清空状态栏
        self.canvas.bind("<Leave>", self.on_mouse_leave)
        
        # Bind Ctrl+Z for undo
        self.root.bind("<Control-z>", lambda event: self.undo())

    def set_mode(self):
        self.mode = self.mode_var.get()

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
            # Limit history to prevent memory issues
            if len(self.history) > 20:
                self.history.pop(0)

    def choose_color(self):
        color = colorchooser.askcolor(title="Choose Drawing Color", color=self.rgb_to_hex(self.draw_color))
        if color[1]: # color is ((r, g, b), "#hex")
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
            self.history = [] # Clear history on new image
            self.scale = 1.0  # Reset scale
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

        # Scale image for display
        w, h = self.original_image.size
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)
        
        # Use Nearest Neighbor to keep pixel edges sharp
        self.display_image = self.original_image.resize((new_w, new_h), Image.NEAREST)
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def get_image_coords(self, event):
        # Convert canvas coordinates back to original image coordinates
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        return int(cx / self.scale), int(cy / self.scale)

    def on_mouse_move(self, event):
        """更新状态栏中的坐标和颜色信息"""
        if self.original_image is None:
            return

        x, y = self.get_image_coords(event)
        w, h = self.original_image.size

        # 更新坐标显示（即使在图像外也显示坐标）
        if self.status_coord_label is not None:
            self.status_coord_label.config(text=f"X:{x}, Y:{y}")

        # 如果在图像范围内，显示像素颜色；否则显示空
        if 0 <= x < w and 0 <= y < h:
            try:
                r, g, b = self.original_image.getpixel((x, y))
                if self.status_color_label is not None:
                    self.status_color_label.config(text=f"B:{b}, G:{g}, R:{r}")
            except Exception:
                if self.status_color_label is not None:
                    self.status_color_label.config(text="B:-, G:-, R:-")
        else:
            if self.status_color_label is not None:
                self.status_color_label.config(text="B:-, G:-, R:-")

    def on_mouse_leave(self, event):
        """鼠标离开画布时清空状态栏"""
        if self.status_coord_label is not None:
            self.status_coord_label.config(text="X:-, Y:-")
        if self.status_color_label is not None:
            self.status_color_label.config(text="B:-, G:-, R:-")

    def on_mouse_down(self, event):
        if self.original_image is None:
            return

        self.save_state() # Save state before modification

        x, y = self.get_image_coords(event)
        
        # Check bounds
        w, h = self.original_image.size
        if x < 0 or y < 0 or x >= w or y >= h:
            return

        if self.mode == "pixel":
            self.paint_pixel(x, y)
        elif self.mode == "rect":
            self.start_x = x
            self.start_y = y
            # Create a temporary rectangle on canvas for visual feedback (scaled coords)
            sx, sy = x * self.scale, y * self.scale
            self.rect_id = self.canvas.create_rectangle(sx, sy, sx, sy, outline="blue", width=2)

    def on_mouse_drag(self, event):
        if self.original_image is None:
            return

        x, y = self.get_image_coords(event)
        w, h = self.original_image.size
        
        # Clamp coordinates
        x = max(0, min(w-1, x))
        y = max(0, min(h-1, y))

        if self.mode == "pixel":
            self.paint_pixel(x, y)
        elif self.mode == "rect" and self.rect_id:
            # Update visual rect (scaled coords)
            sx, sy = self.start_x * self.scale, self.start_y * self.scale
            ex, ey = (x + 1) * self.scale, (y + 1) * self.scale
            self.canvas.coords(self.rect_id, sx, sy, ex, ey)

    def on_mouse_up(self, event):
        if self.mode == "rect" and self.rect_id:
            x, y = self.get_image_coords(event)
            w, h = self.original_image.size
            x = max(0, min(w-1, x))
            y = max(0, min(h-1, y))

            # Finalize rectangle on the actual image
            draw = ImageDraw.Draw(self.original_image)
            # Ensure coordinates are top-left to bottom-right
            x0, y0 = min(self.start_x, x), min(self.start_y, y)
            x1, y1 = max(self.start_x, x), max(self.start_y, y)
            
            draw.rectangle([x0, y0, x1, y1], fill=self.draw_color, outline=None)
            
            # Remove UI rectangle and refresh image
            self.canvas.delete(self.rect_id)
            self.rect_id = None
            self.start_x = None
            self.start_y = None
            self.refresh_canvas()

    def paint_pixel(self, x, y):
        # Direct pixel access is faster than ImageDraw for single pixels
        try:
            self.original_image.putpixel((x, y), self.draw_color)
            # Optimization: Draw a small rectangle on canvas to mimic the change immediately
            # Must account for scale
            sx, sy = x * self.scale, y * self.scale
            self.canvas.create_rectangle(sx, sy, sx + self.scale, sy + self.scale, 
                                       outline="", fill=self.rgb_to_hex(self.draw_color))
        except IndexError:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = MapEditorApp(root)
    root.mainloop()