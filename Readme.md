# Basic Photo Editor – Digital Image‑Processing

A stand‑alone Python desktop app that edits and adds effectst to photos based on algorithms learnt in Digital Image Processing Course.

| Category         | Tools included                                       |
| ---------------- | ---------------------------------------------------- |
| Basic transforms | **Flip** (H/V) · **Rotate** (any angle, auto‑expand) |
| Tuning           | **Brightness / Contrast** sliders                    |
| Blur             | **Gaussian blur** (custom kernel)                    |
| Special effects  | **Chroma‑key** composite                             |
| Geometric        | **Radial undistort** (`k` coefficient)               |
| Frequency        | **Ideal low‑pass** filter (radius‑controlled)        |

The GUI also has **Open**, **Save**, and **Reset** buttons plus a 500 × 500 live preview.

---

## Requirements

-   Python 3.8+
-   `opencv‑python`
-   `numpy`
-   `pillow` (PIL fork for GUI preview)

Install:

```bash
pip install opencv-python numpy pillow
```

## Usage

### Command‑line usage

The general pattern is:

    python3 photo_editor.py <input> <output> <command> [options]

---

#### 1. flip

Flip an image horizontally or vertically.

-   **Option**
    -   --direction {horizontal,vertical} (default horizontal)

Example:

    python3 photo_editor.py in.jpg out.jpg flip --direction vertical

---

#### 2. rotate

Rotate by an arbitrary angle (degrees).

-   **Argument**
    -   <angle> (float, positive = counter‑clockwise)

Example:

    python3 photo_editor.py in.jpg out.jpg rotate 45

---

#### 3. bc (brightness / contrast)

Adjust brightness and contrast.

-   **Arguments**
    -   <brightness> (‑100 … 100)
    -   <contrast> (0.2 … 3.0)

Example:

    python3 photo_editor.py in.jpg out.jpg bc 20 1.3

---

#### 4. blur

Apply Gaussian blur.

-   **Option**
    -   --kernel <odd int> (e.g. 3, 5, 7)

Example:

    python3 photo_editor.py in.jpg out.jpg blur --kernel 7

---

#### 5. chroma

Perform green‑screen composite.

-   **Options**
    -   --background <bg.jpg> (required)
    -   --threshold <int> (default 60)

Example:

    python3 photo_editor.py fg.png keyed.png chroma --background bg.jpg --threshold 50

---

#### 6. undistort

Correct radial barrel or pincushion distortion.

-   **Option**
    -   --k <float> (positive → barrel, negative → pincushion)

Example:

    python3 photo_editor.py in.jpg out.jpg undistort --k 0.0005

---

#### 7. lowpass

Ideal low‑pass filter in the frequency domain.

-   **Option**
    -   --radius <int> (cut‑off in pixels)

Example:

    python3 photo_editor.py in.jpg out.jpg lowpass --radius 35

---

### Launch the GUI

    python3 photo_editor.py --gui

Running the script with no arguments opens the GUI as well.
