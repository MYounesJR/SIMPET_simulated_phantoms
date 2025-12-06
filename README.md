# SIMPET-3D: Simulated 3D PET Phantom

This repository provides a **simple Python script** to generate 3D *PET-like* phantoms:

- a 3D **emission** volume with a rotated ellipsoid and random inner structures  
- a corresponding **attenuation** map  
- outputs saved both as:
  - **NumPy arrays** (`.npy`)
  - **Interfile** raw images (`.img` + `.hdr`, float32, X-fastest)

Although it was designed for PET, the volumes are generic 3D images and can be reused for:
- testing 3D reconstruction algorithms
- validating registration / segmentation methods
- deep learning experiments with 3D data (denoising, super-resolution, etc.)

---

## 1. Files in this repo

- `simpet3d.py` – main script (can be run from the terminal or imported as a module)
- `README.md` – this documentation
- `requirements.txt` – minimal Python dependencies

---

## 2. Installation

### 2.1. Create and activate an environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
# or:
venv\Scripts\activate             # Windows
