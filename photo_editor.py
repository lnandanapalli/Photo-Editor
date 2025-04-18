#!/usr/bin/env python3
from __future__ import annotations
import cv2, numpy as np, argparse, sys, tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk


def flip_image(img: np.ndarray, direction: str = "horizontal") -> np.ndarray:
    return img[:, ::-1] if direction == "horizontal" else img[::-1]


def rotate_image(img: np.ndarray, angle: float, expand: bool = True) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    if expand:
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += new_w / 2 - w / 2
        M[1, 2] += new_h / 2 - h / 2
        w, h = new_w, new_h
    return cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def adjust_brightness_contrast(
    img: np.ndarray, brightness: int = 0, contrast: float = 1.0
) -> np.ndarray:
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


def gaussian_blur(img: np.ndarray, k: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(img, (k, k), 0)


def chroma_key(
    fg: np.ndarray,
    bg: np.ndarray,
    key_rgb: tuple[int, int, int] = (0, 255, 0),
    thr: int = 60,
) -> np.ndarray:
    if bg.shape[:2] != fg.shape[:2]:
        bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]))
    diff = cv2.absdiff(fg.astype(np.int16), np.array(key_rgb, np.int16))
    mask = (diff.sum(axis=2) < thr).astype(np.uint8)
    mask3 = cv2.merge([mask] * 3)
    return np.where(mask3 == 1, bg, fg)


def radial_undistort(img: np.ndarray, k: float, interp: str = "bilinear") -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    yy, xx = np.indices((h, w), np.float32)
    x, y = xx - cx, yy - cy
    r = np.sqrt(x * x + y * y)
    factor = 1 - k * r
    map_x = (x * factor + cx).astype(np.float32)
    map_y = (y * factor + cy).astype(np.float32)
    flag = cv2.INTER_LINEAR if interp == "bilinear" else cv2.INTER_NEAREST
    return cv2.remap(img, map_x, map_y, flag)


def ideal_low_pass(img_g: np.ndarray, radius: int = 40) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(img_g))
    h, w = img_g.shape
    y, x = np.ogrid[:h, :w]
    mask = ((x - w // 2) ** 2 + (y - h // 2) ** 2) <= radius * radius
    filtered = f * mask
    inv = np.fft.ifft2(np.fft.ifftshift(filtered))
    return np.abs(inv).clip(0, 255).astype(np.uint8)


class PhotoEditorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Photo Editor")

        # Preview area
        preview_container = tk.Frame(self.root, bg="#2e2e2e")
        preview_container.pack(side="left", fill="both", expand=True)
        self.preview = tk.Label(preview_container, bg="#2e2e2e")
        self.preview.pack(anchor="center", expand=True)

        # Control pane
        pane = tk.Frame(self.root)
        pane.pack(side="right", fill="y", padx=6)
        ttk.Style().configure("TButton", padding=4)

        ttk.Button(pane, text="Open", command=self.open_file).pack(fill="x")
        ttk.Button(pane, text="Save", command=self.save_file).pack(fill="x")
        ttk.Button(pane, text="Reset", command=self.reset_preview).pack(fill="x")
        ttk.Separator(pane).pack(fill="x", pady=4)

        ttk.Label(pane, text="Operation").pack(anchor="w")
        self.op = ttk.Combobox(
            pane,
            state="readonly",
            values=[
                "flip",
                "rotate",
                "brightness/contrast",
                "blur",
                "chroma",
                "undistort",
                "low‑pass",
            ],
        )
        self.op.current(0)
        self.op.pack(fill="x")

        self.param_frame = tk.Frame(pane)
        self.param_frame.pack(fill="x", pady=4)
        self.op.bind("<<ComboboxSelected>>", lambda e: self.build_params())
        self.build_params()

        ttk.Button(pane, text="Apply", command=self.apply).pack(fill="x", pady=6)

        self.img_bgr: np.ndarray | None = None
        self.res_bgr: np.ndarray | None = None

    def build_params(self):
        for w in self.param_frame.winfo_children():
            w.destroy()
        op = self.op.get()
        if op == "flip":
            self.dir = tk.StringVar(value="horizontal")
            for val in ("horizontal", "vertical"):
                ttk.Radiobutton(
                    self.param_frame, text=val, variable=self.dir, value=val
                ).pack(anchor="w")
        elif op == "rotate":
            ttk.Label(self.param_frame, text="Angle°").pack(anchor="w")
            self.angle = tk.DoubleVar(value=90)
            ttk.Entry(self.param_frame, textvariable=self.angle).pack(fill="x")
        elif op == "brightness/contrast":
            ttk.Label(self.param_frame, text="Brightness (-100→100)").pack(anchor="w")
            self.bright = tk.IntVar(value=0)
            ttk.Scale(
                self.param_frame,
                from_=-100,
                to=100,
                orient="horizontal",
                variable=self.bright,
            ).pack(fill="x")
            ttk.Label(self.param_frame, text="Contrast (0.2→3.0)").pack(anchor="w")
            self.contrast = tk.DoubleVar(value=1.0)
            ttk.Scale(
                self.param_frame,
                from_=0.2,
                to=3.0,
                orient="horizontal",
                variable=self.contrast,
            ).pack(fill="x")
        elif op == "blur":
            ttk.Label(self.param_frame, text="Kernel (odd)").pack(anchor="w")
            self.kb = tk.IntVar(value=5)
            ttk.Entry(self.param_frame, textvariable=self.kb).pack(fill="x")
        elif op == "chroma":
            ttk.Label(self.param_frame, text="Background").pack(anchor="w")
            self.bg_path = ttk.Entry(self.param_frame)
            self.bg_path.pack(fill="x")
            ttk.Button(self.param_frame, text="Browse", command=self.browse_bg).pack()
            ttk.Label(self.param_frame, text="Threshold").pack(anchor="w")
            self.ck_thr = tk.IntVar(value=60)
            ttk.Entry(self.param_frame, textvariable=self.ck_thr).pack(fill="x")
        elif op == "undistort":
            ttk.Label(self.param_frame, text="k coeff").pack(anchor="w")
            self.k_var = tk.DoubleVar(value=0.0005)
            ttk.Entry(self.param_frame, textvariable=self.k_var).pack(fill="x")
        elif op == "low‑pass":
            ttk.Label(self.param_frame, text="Radius").pack(anchor="w")
            self.rad = tk.IntVar(value=40)
            ttk.Entry(self.param_frame, textvariable=self.rad).pack(fill="x")

    def open_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if not path:
            return
        self.img_bgr = cv2.imread(path)
        self.res_bgr = None
        self.show(self.img_bgr)

    def save_file(self):
        if self.res_bgr is None:
            messagebox.showwarning("Nothing", "No result to save")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg")],
        )
        if path:
            cv2.imwrite(path, self.res_bgr)

    def reset_preview(self):
        if self.img_bgr is not None:
            self.res_bgr = None
            self.show(self.img_bgr)

    def browse_bg(self):
        p = filedialog.askopenfilename(
            filetypes=[("Image", "*.png;*.jpg;*.jpeg;*.bmp")]
        )
        if p:
            self.bg_path.delete(0, "end")
            self.bg_path.insert(0, p)

    def show(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((500, 500))
        imgtk = ImageTk.PhotoImage(pil)
        self.preview.configure(image=imgtk)
        self.preview.image = imgtk

    def apply(self):
        if self.img_bgr is None:
            messagebox.showwarning("No image", "Open image first")
            return
        op = self.op.get()
        img = self.img_bgr.copy()
        try:
            if op == "flip":
                self.res_bgr = flip_image(img, self.dir.get())
            elif op == "rotate":
                self.res_bgr = rotate_image(img, self.angle.get())
            elif op == "brightness/contrast":
                self.res_bgr = adjust_brightness_contrast(
                    img, self.bright.get(), self.contrast.get()
                )
            elif op == "blur":
                self.res_bgr = gaussian_blur(img, self.kb.get() | 1)
            elif op == "chroma":
                bg = cv2.imread(self.bg_path.get())
                self.res_bgr = chroma_key(img, bg, thr=self.ck_thr.get())
            elif op == "undistort":
                self.res_bgr = radial_undistort(img, self.k_var.get())
            elif op == "low‑pass":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                res = ideal_low_pass(gray, self.rad.get())
                self.res_bgr = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
            else:
                messagebox.showinfo("TODO", op)
                return
            self.show(self.res_bgr)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run(self):
        self.root.mainloop()


def main(argv):
    if "--gui" in argv or len(argv) == 0:
        PhotoEditorGUI().run()
        return
    if "--help-cli" in argv:
        print(
            "CLI pattern: photo_editor.py input output <cmd> [options]\n"
            "Commands: flip rotate bc blur chroma undistort lowpass"
        )
        return
    inp, out, cmd, *rest = argv
    img = cv2.imread(inp)
    out_bgr = None
    if cmd == "flip":
        direction = (
            rest[rest.index("--direction") + 1]
            if "--direction" in rest
            else "horizontal"
        )
        out_bgr = flip_image(img, direction)
    elif cmd == "rotate":
        angle = float(rest[0])
        out_bgr = rotate_image(img, angle)
    elif cmd == "bc":
        br = int(rest[0])
        ct = float(rest[1])
        out_bgr = adjust_brightness_contrast(img, br, ct)
    else:
        sys.exit("Unknown CLI command")
    cv2.imwrite(out, out_bgr)


if __name__ == "__main__":
    main(sys.argv[1:])
