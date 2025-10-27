# 🌌 Image Resampling and Interpolation — From First Principles to Advanced Methods

> 🎯 *A comprehensive theoretical and practical guide for researchers, students, and practitioners in satellite imaging, computer vision, and image processing.*

---

## 🧭 Table of Contents

1. [Introduction](#-introduction)
2. [Why Resampling Matters (Especially in Satellite Imagery)](#-why-resampling-matters-especially-in-satellite-imagery)
3. [Mathematical Foundations](#-mathematical-foundations)
4. [Interpolation Methods](#-interpolation-methods)
    - [1️⃣ Nearest Neighbor Interpolation](#1️⃣-nearest-neighbor-interpolation)
    - [2️⃣ Bilinear Interpolation](#2️⃣-bilinear-interpolation)
    - [3️⃣ Bicubic Interpolation](#3️⃣-bicubic-interpolation)
    - [4️⃣ Spline Interpolation](#4️⃣-spline-interpolation)
    - [5️⃣ Lanczos Resampling](#5️⃣-lanczos-resampling)
    - [6️⃣ Area Interpolation](#6️⃣-area-interpolation)
    - [7️⃣ Pixel Shuffle (Sub-Pixel Convolution)](#7️⃣-pixel-shuffle-sub-pixel-convolution)
    - [8️⃣ Adaptive Average & Max Pooling](#8️⃣-adaptive-average--max-pooling)
    - [9️⃣ Advanced / Learnable Interpolation](#9️⃣-advanced--learnable-interpolation)
5. [🧠 Kernel Intuition & Visualization](#-kernel-intuition--visualization)
6. [🧰 PyTorch Implementations](#-pytorch-implementations)
7. [🌍 Domain-Specific Applications](#-domain-specific-applications)
8. [📊 Summary Comparison Table](#-summary-comparison-table)
9. [📚 References](#-references)

---

## 🌅 Introduction

Image resampling, or interpolation, is the process of reconstructing or estimating pixel values when transforming an image — for example, resizing, rotating, reprojecting, or aligning multi-resolution data.

From a mathematical viewpoint, **an image is a discrete sampling of a continuous spatial function**:

```math
I(x, y): \mathbb{Z}^2 \rightarrow \mathbb{R}^n
```


Resampling estimates the intensity $ I'(x', y') $ at new spatial locations that may not align with the original pixel grid.

---

## 🌍 Why Resampling Matters (Especially in Satellite Imagery)

In **remote sensing**, resampling is a critical preprocessing step:

| Task | Description | Example |
|------|--------------|----------|
| 🛰️ **Geometric Correction** | Aligning images from different sensors or dates | Registering Landsat and Sentinel-2 scenes |
| 🌐 **Reprojection** | Transforming data between coordinate systems | WGS84 → UTM |
| 🧩 **Mosaicking** | Stitching adjacent tiles seamlessly | Large-area orthomosaics |
| 🧠 **Super-Resolution / Fusion** | Combining multiple sources for higher detail | Pansharpening, data fusion |
| 🧾 **Downscaling / Averaging** | Reducing resolution while preserving radiometry | Cloud-free composites |

Resampling affects *radiometric accuracy*, *geometric fidelity*, and *scientific interpretability*, making the choice of method crucial.

---

## ⚙️ Mathematical Foundations

Let’s derive resampling **from first principles**.

We view a digital image as samples of a continuous signal $ f(x, y) $ at discrete integer coordinates:

```math
I(i, j) = f(i, j)
```

To obtain the image at new coordinates $(x', y')$, we reconstruct the continuous function using an **interpolation kernel** $ h(x, y) $:

```math
f(x', y') = \sum_i \sum_j I(i, j) \, h(x' - i, y' - j)
```


Then, the resampled image is obtained as:
```math
I'(x', y') = f(x', y')
```


The **kernel $ h(x, y) $** defines how nearby pixels influence the interpolated value — from simple nearest-pixel selection to complex, smooth polynomial blending.

---

## 🔢 Interpolation Methods

Each method uses a different kernel function $ h(x) $.  
We’ll start from the simplest (Nearest Neighbor) and move toward the most advanced (Learnable Interpolation).

---

### 1️⃣ Nearest Neighbor Interpolation

<details>
<summary>🧩 Click to expand</summary>

#### **Theory**
For each new pixel, assign the value of the nearest pixel from the input image:

```math
I'(x', y') = I(\text{round}(x'), \text{round}(y'))
```


#### **Intuition**
- Conceptually simple: “pick the closest pixel.”
- Fast, no averaging or smoothing.
- Produces *blocky* or *aliased* edges.

#### **PyTorch Example**
```python
import torch
import torchvision.transforms.functional as F
from PIL import Image

img = Image.open("satellite_image.jp2")
img_tensor = F.to_tensor(img).unsqueeze(0)
resampled = torch.nn.functional.interpolate(img_tensor, scale_factor=2, mode='nearest')
````

#### **Applications**

* Label masks (e.g., land cover classification)
* Fast previews or visualization
* Non-continuous data (categorical)

#### **Characteristics**

| Smoothness | Differentiable | Speed  | Artifacts |
| ---------- | -------------- | ------ | --------- |
| ❌ Low      | ✅              | ⚡ Fast | Blocky    |

</details>

---

### 2️⃣ Bilinear Interpolation

<details>
<summary>🧩 Click to expand</summary>

#### **Theory**

Linear interpolation along x and y axes.

**1D kernel:**
```math
h(x) = \max(1 - |x|, 0)
```


**2D separable form:**
```math
I'(x', y') = \sum_i \sum_j I(i, j) , h(x' - i) , h(y' - j)
```


This means each interpolated value is a **weighted average of the four nearest neighbors**.

#### **Intuition**

Smooth transitions, but not perfectly sharp.
Balances simplicity and quality.

#### **PyTorch Example**

```python
resampled = torch.nn.functional.interpolate(
    img_tensor, scale_factor=2, mode='bilinear', align_corners=True)
```

#### **Applications**

* Common default in GIS and remote sensing reprojections
* General image resizing

#### **Characteristics**

| Smoothness | Differentiable | Speed | Visual Quality |
| ---------- | -------------- | ----- | -------------- |
| ✅ Moderate | ✅              | ✅     | 👍 Good        |

</details>

---

### 3️⃣ Bicubic Interpolation

<details>
<summary>🧩 Click to expand</summary>

#### **Mathematical Derivation**

The **cubic convolution kernel** $ h(x) $ (Keys, 1981):

```math
h(x) =
\begin{cases}
(1.5)|x|^3 - 2.5|x|^2 + 1, & |x| < 1 \
-0.5|x|^3 + 2.5|x|^2 - 4|x| + 2, & 1 \le |x| < 2 \
0, & |x| \ge 2
\end{cases}
```


For 2D:
```math
I'(x', y') = \sum_i \sum_j I(i, j) , h(x' - i) , h(y' - j)
```

#### **Intuition**

* Uses 16 nearest pixels (4×4 neighborhood)
* Produces smoother, sharper results than bilinear
* Preserves edge details and gradient continuity

#### **Applications**

* High-quality satellite imagery enhancement
* Map projections, orthorectification

#### **Characteristics**

| Smoothness | Differentiable | Speed       | Edge Preservation |
| ---------- | -------------- | ----------- | ----------------- |
| ✅✅ High    | ✅              | ⚠️ Moderate | 💎 Excellent      |

</details>

---

### 4️⃣ Spline Interpolation

<details>
<summary>🧩 Click to expand</summary>

#### **Theory**

Spline interpolation uses **piecewise polynomial functions** to ensure smoothness up to the $ n^{th} $ derivative.

A general B-spline interpolation:

```math
I'(x') = \sum_i I(i) , B_n(x' - i)
```


where $ B_n $ is an $ n^{th} $-order basis spline.

#### **Intuition**

* Produces globally smooth surfaces.
* Higher order = smoother but risk of over-smoothing.

#### **Applications**

* Scientific and geospatial surface fitting
* DEM resampling and continuous field interpolation

</details>

---

### 5️⃣ Lanczos Resampling

<details>
<summary>🧩 Click to expand</summary>

#### **Kernel**

```math
h(x) = \text{sinc}(x) , \text{sinc}\left(\frac{x}{a}\right), \quad |x| < a
```


#### **Explanation**

Approximates an *ideal sinc interpolation*, which perfectly reconstructs a band-limited signal.

#### **Intuition**

* Excellent frequency preservation.
* Reduces aliasing during downsampling.

#### **Applications**

* Precision satellite imagery
* Downscaling high-frequency data (e.g., urban textures)

</details>

---

### 6️⃣ Area Interpolation

<details>
<summary>🧩 Click to expand</summary>

#### **Theory**

For downsampling, compute the **area-weighted average** of input pixels that overlap with each output pixel.

```math
I'(x', y') = \frac{\sum I(x, y) \cdot A(x, y)}{\sum A(x, y)}
```


where $ A(x, y) $ is the overlap area.

#### **Intuition**

Preserves *energy* (radiometric integrity) rather than sharpness.

#### **Applications**

* Radiometric resampling in remote sensing
* Downsampling reflectance products (Sentinel-2, MODIS)

</details>

---

### 7️⃣ Pixel Shuffle (Sub-Pixel Convolution)

<details>
<summary>🧩 Click to expand</summary>

Introduced by **Shi et al., CVPR 2016**
*Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel CNN*

#### **Idea**

Rearrange feature channels into higher spatial resolution — a *learned interpolation*.

#### **Mathematics**

For a scale factor ( r ),
Input shape: `[B, C×r², H, W]` → Output shape: `[B, C, H×r, W×r]`.

```math
I'*{c, h, w} = I*{c \times r^2 + (h \bmod r) \times r + (w \bmod r), \lfloor h / r \rfloor, \lfloor w / r \rfloor}
```


#### **PyTorch Example**

```python
import torch.nn as nn
ps = nn.PixelShuffle(upscale_factor=2)
output = ps(feature_map)
```

#### **Applications**

* Deep learning super-resolution (SRGAN, ESRGAN, etc.)
* Learned interpolation for satellite SR

</details>

---

### 8️⃣ Adaptive Average & Max Pooling

<details>
<summary>🧩 Click to expand</summary>

#### **Concept**

Downsamples to a *fixed output size* regardless of input dimensions.

```math
y_{ij} = \frac{1}{N_{ij}} \sum_{p,q \in R_{ij}} x_{pq}
```


* **AdaptiveAvgPool** → smooth, energy-preserving
* **AdaptiveMaxPool** → feature-selective, edge-preserving

#### **Applications**

* CNN-based satellite feature extraction
* Multi-scale fusion networks

</details>

---

### 9️⃣ Advanced / Learnable Interpolation

<details>
<summary>🧩 Click to expand</summary>

Includes **data-driven interpolation** methods:

| Method                         | Description                                           |
| ------------------------------ | ----------------------------------------------------- |
| **Deformable Convolutions**    | Learn adaptive offsets for sampling locations         |
| **MetaSR / LIIF**              | Learn continuous, content-aware interpolation kernels |
| **Diffusion-based Resampling** | Denoising diffusion models for realistic upscaling    |
| **Transformer Interpolation**  | Context-aware super-resolution in geospatial vision   |

**Mathematical form:**
```math
I'(x', y') = \sum_{i,j} w_{ij}(x', y') , I(x+i, y+j)
```

where weights $ w_{ij} $ are *learned functions of input content*.

</details>

---

## 🧠 Kernel Intuition & Visualization

| Method   | Kernel Shape | Description               |
| -------- | ------------ | ------------------------- |
| Nearest  | ⬜            | Step (box) function       |
| Bilinear | 🔺           | Triangular                |
| Bicubic  | 🌀           | Smooth polynomial         |
| Lanczos  | 🔊           | Sinc-windowed oscillatory |
| Area     | ▧            | Weighted overlap region   |

You can visualize these kernels using Python:

```python
import numpy as np, matplotlib.pyplot as plt

x = np.linspace(-3, 3, 1000)
triangular = np.maximum(1 - np.abs(x), 0)

plt.plot(x, triangular)
plt.title("Bilinear Interpolation Kernel")
plt.grid()
plt.show()
```

---

## 🧰 PyTorch Implementations

Demonstrating on `.tif`, `.jp2`, and `.png` satellite images:

```python
import torch
import rasterio
from PIL import Image
import torchvision.transforms.functional as F

# Example: Load JP2 or TIF
with rasterio.open("sentinel_image.tif") as src:
    img = src.read([1,2,3])
    img = torch.tensor(img / img.max()).unsqueeze(0)

# Apply interpolation
res_bilinear = torch.nn.functional.interpolate(img, scale_factor=2, mode='bilinear', align_corners=True)
res_lanczos = torch.nn.functional.interpolate(img, scale_factor=2, mode='bicubic')
```

---

## 🌍 Domain-Specific Applications

| Domain                    | Task                     | Recommended Methods              |
| ------------------------- | ------------------------ | -------------------------------- |
| **Remote Sensing**        | Reprojection, mosaicking | Bilinear, Bicubic, Lanczos       |
| **Super-Resolution**      | Deep learning SR         | Pixel Shuffle, MetaSR            |
| **GIS Visualization**     | Downsampling             | Area interpolation               |
| **DEM Processing**        | Continuous surfaces      | B-spline, Bicubic                |
| **Hyperspectral Imaging** | Spectral-spatial SR      | Deformable or Meta-interpolation |

---

## 📊 Summary Comparison Table

| Method           | Smoothness | Speed | Use-Case     | Differentiable | Common in RS |
| ---------------- | ---------- | ----- | ------------ | -------------- | ------------ |
| Nearest          | Low        | ⚡     | Masks        | ✅              | ✅            |
| Bilinear         | Medium     | ✅     | General      | ✅              | ✅            |
| Bicubic          | High       | ⚠️    | Enhancement  | ✅              | ✅            |
| Lanczos          | Very High  | ❌     | Precision    | ✅              | ✅            |
| Area             | High       | ⚠️    | Downsampling | ✅              | ✅            |
| Pixel Shuffle    | Learnable  | ⚠️    | SR           | ✅              | ✅            |
| Adaptive Pooling | Medium     | ✅     | CNNs         | ✅              | ✅            |

---

## 📚 References

1. Gonzalez & Woods, *Digital Image Processing*, 4th Edition (2018): [Book Link](https://www.cl72.org/090imagePLib/books/Gonzales,Woods-Digital.Image.Processing.4th.Edition.pdf)
2. R. G. Keys, *Cubic Convolution Interpolation for Digital Image Processing*, IEEE Trans. ASSP, 1981: [Paper Link](https://www.ncorr.com/download/publications/keysbicubic.pdf)
3. Szeliski, *Computer Vision: Algorithms and Applications*, Springer (2022): [Book Link](https://library.huree.edu.mn/data/202295/2024-06-03/Computer%20Vision%20-%20Algorithms%20and%20Applications%202nd%20Edition,%20Richard%20Szeliski.pdf)
4. Shi et al., *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel CNN*, CVPR 2016: [Paper Link](https://arxiv.org/abs/1609.05158)
5. GDAL Docs: [Resampling Overview](https://gdal.org/programs/gdalwarp.html#resampling-methods)
6. Wikipedia: [Resampling (Image Scalinging)](https://en.wikipedia.org/wiki/Image_scaling)
7. PyTorch Docs: [torch.nn.functional.interpolate](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html)
   
---

> 💡 *This guide aims to make interpolation both intuitive and rigorous — blending signal processing, satellite imaging, and deep learning theory in one place.*

---
