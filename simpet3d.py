#!/usr/bin/env python3
"""
SIMPET 3D phantom generator

- Generates a 3D “PET-like” emission volume with a rotated ellipsoid + inner structures.
- Optionally applies a 3D elastic deformation.
- Builds a binary-like attenuation map from the emission volume.
- Saves:
    * emission.npy, attenuation.npy
    * Interfile emission   (.img + .hdr, float32)
    * Interfile attenuation(.img + .hdr, float32)

Run from terminal, for example:
    python simpet3d.py --out_dir ./SIMPET_samples

You can also import it from Python:
    import simpet3d
    result = simpet3d.generate_simpet_sample(out_dir="./SIMPET_samples")
"""

import os
import argparse
import random

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

# Try to use elasticdeform if present (better quality & faster)
try:
    import elasticdeform as _ed
    _HAS_ED = True
except Exception:
    _HAS_ED = False


# -----------------------
# Rotation helpers
# -----------------------
def euler_xyz_to_R(rx, ry, rz):
    """R = Rz * Ry * Rx; angles in radians."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx,  cx]], dtype=np.float32)
    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


# -----------------------
# Draw rotated ellipsoid in a local bbox
# -----------------------
def add_rotated_ellipsoid(vol, center, axes, angles_deg, value):
    """
    Fill a rotated ellipsoid:
      (x')^2/ax^2 + (y')^2/ay^2 + (z')^2/az^2 <= 1
    where [x',y',z'] = R^T * ([x,y,z] - c).

    vol: 3D array [Z,Y,X]
    center: (zc, yc, xc) in voxel indices
    axes: (az, ay, ax) semi-axes lengths
    angles_deg: (rx, ry, rz) in degrees
    value: float to assign inside ellipsoid
    """
    zc, yc, xc = center
    az, ay, ax = axes  # semi-axes along (z,y,x) in the ellipsoid frame
    D, H, W = vol.shape
    rmax = int(np.ceil(max(ax, ay, az)))  # conservative bbox radius per axis
    z0, z1 = max(0, zc - rmax), min(D, zc + rmax + 1)
    y0, y1 = max(0, yc - rmax), min(H, yc + rmax + 1)
    x0, x1 = max(0, xc - rmax), min(W, xc + rmax + 1)

    # local grid (Z,Y,X) relative to center
    Z, Y, X = np.meshgrid(
        np.arange(z0, z1, dtype=np.float32) - zc,
        np.arange(y0, y1, dtype=np.float32) - yc,
        np.arange(x0, x1, dtype=np.float32) - xc,
        indexing='ij'
    )
    P = np.stack((Z, Y, X), axis=0).reshape(3, -1)

    rx, ry, rz = np.deg2rad(angles_deg)
    R = euler_xyz_to_R(rx, ry, rz)  # world->ellipsoid: use R^T
    Q = (R.T @ P).reshape(3, Z.shape[0], Z.shape[1], Z.shape[2])
    Zr, Yr, Xr = Q[0], Q[1], Q[2]

    mask = (Xr/ax)**2 + (Yr/ay)**2 + (Zr/az)**2 <= 1.0
    sub = vol[z0:z1, y0:y1, x0:x1]
    sub[mask] = value
    vol[z0:z1, y0:y1, x0:x1] = sub


# -----------------------
# Inside tests (rotated parent ellipsoid)
# -----------------------
def normalized_radius_in_parent(point, parent_center, parent_axes, parent_angles_deg):
    """Return t in [0,∞): t<1 means the point is inside the rotated parent ellipsoid."""
    z, y, x = point
    cz, cy, cx = parent_center
    Az, Ay, Ax = parent_axes
    rx, ry, rz = np.deg2rad(parent_angles_deg)
    R = euler_xyz_to_R(rx, ry, rz)
    p = np.array([z - cz, y - cy, x - cx], dtype=np.float32)
    pr = R.T @ p
    t = np.sqrt((pr[2]/Ax)**2 + (pr[1]/Ay)**2 + (pr[0]/Az)**2)  # note: axes order (Zr,Yr,Xr)
    return float(t)


def child_ellipsoid_fits(parent_center, parent_axes, parent_angles_deg,
                         child_center, child_axes):
    """
    Conservative sufficient condition:
      let t = normalized radius of child_center in parent (t<1 => center inside)
      remaining margin in parent "u-space" ~ (1 - t)
      ensure max(child_axes) <= (1 - t) * min(parent_axes)
    This guarantees containment for any child rotation (conservative).
    """
    t = normalized_radius_in_parent(child_center, parent_center, parent_axes, parent_angles_deg)
    if t >= 1.0:
        return False
    margin_units = (1.0 - t) * min(parent_axes)  # strict but safe
    return max(child_axes) <= margin_units


# -----------------------
# Phantom generator
# -----------------------
def generate_phantom3d(
    shape=(400, 400, 400),
    center_Amax=160,
    center_ratio_range=(0.6, 0.8),
    center_intensity=20.0,
    n_structures_range=(10, 50),
    size_range=(5, 30),
    ellipsoid_prob=0.6,
    child_intensity_range=(5.0, 400.0),
    parent_rot_range_deg=(0.0, 180.0),
    child_rot_range_deg=(0.0, 180.0),
    max_trials_per_child=300,
    seed=None,
):
    """
    Returns volume float32:
      - background 0
      - central *rotated ellipsoid* (intensity=center_intensity),
        axes with ratios in [center_ratio_range]
      - small objects inside parent, random rotations/positions/intensities/sizes
    """
    rng = np.random.default_rng(seed)
    D, H, W = shape
    vol = np.zeros(shape, dtype=np.float32)

    # --- Central rotated ellipsoid axes with ratios in [0.6,0.8]
    rmin, rmax = center_ratio_range
    # draw two ratios independently so min/max ∈ [0.6,0.8]
    r1 = float(rng.uniform(rmin, rmax))
    r2 = float(rng.uniform(rmin, rmax))
    # randomly permute which axis is max
    order = rng.permutation(3)
    axes_list = np.array([center_Amax, int(center_Amax*r1), int(center_Amax*r2)], dtype=np.int32)
    Az, Ay, Ax = axes_list[order]
    parent_axes = (int(Az), int(Ay), int(Ax))

    # parent rotation
    ang_lo, ang_hi = parent_rot_range_deg
    parent_angles = (float(rng.uniform(ang_lo, ang_hi)),
                     float(rng.uniform(ang_lo, ang_hi)),
                     float(rng.uniform(ang_lo, ang_hi)))
    parent_center = (D//2, H//2, W//2)

    # draw central ellipsoid
    add_rotated_ellipsoid(vol, parent_center, parent_axes, parent_angles,
                          value=float(center_intensity))

    # --- Small objects
    n_small = int(rng.integers(n_structures_range[0], n_structures_range[1] + 1))
    smin, smax = size_range
    ilo, ihi = child_intensity_range

    placed = 0
    trials = 0
    while placed < n_small and trials < n_small * max_trials_per_child:
        trials += 1

        # type + dimensions
        is_ellip = (rng.random() < ellipsoid_prob)
        if is_ellip:
            ax_e = int(rng.integers(smin, smax + 1))
            ay_e = int(rng.integers(smin, smax + 1))
            az_e = int(rng.integers(smin, smax + 1))
            child_axes = (az_e, ay_e, ax_e)
        else:
            r = int(rng.integers(smin, smax + 1))
            child_axes = (r, r, r)

        # choose a candidate center within parent bbox range, then accept if fits
        # sample around parent center within +/- min(parent_axes)
        zc = int(rng.integers(parent_center[0]-parent_axes[0], parent_center[0]+parent_axes[0]+1))
        yc = int(rng.integers(parent_center[1]-parent_axes[1], parent_center[1]+parent_axes[1]+1))
        xc = int(rng.integers(parent_center[2]-parent_axes[2], parent_center[2]+parent_axes[2]+1))
        child_center = (zc, yc, xc)

        if not child_ellipsoid_fits(parent_center, parent_axes, parent_angles, child_center, child_axes):
            continue  # try again

        # rotation for child
        cang = (float(rng.uniform(*child_rot_range_deg)),
                float(rng.uniform(*child_rot_range_deg)),
                float(rng.uniform(*child_rot_range_deg)))

        intensity = float(rng.uniform(ilo, ihi))

        add_rotated_ellipsoid(vol, child_center, child_axes, cang, intensity)
        placed += 1

    meta = dict(parent_center=parent_center,
                parent_axes=parent_axes,
                parent_angles=parent_angles,
                n_small=placed)
    return vol.astype(np.float32), meta


# -----------------------
# Elastic deformation
# -----------------------
def _elastic_deform_3d_scipy(vol, sigma_grid, alpha, order=3, mode='constant', cval=0.0, seed=None):
    """
    SciPy fallback: random displacement fields smoothed by Gaussian (sigma_grid),
    scaled by alpha, then map_coordinates. vol is float32 [Z,Y,X].
    """
    rng = np.random.default_rng(seed)
    D, H, W = vol.shape
    dx = gaussian_filter(rng.standard_normal((D, H, W)).astype(np.float32), sigma=sigma_grid) * alpha
    dy = gaussian_filter(rng.standard_normal((D, H, W)).astype(np.float32), sigma=sigma_grid) * alpha
    dz = gaussian_filter(rng.standard_normal((D, H, W)).astype(np.float32), sigma=sigma_grid) * alpha
    z, y, x = np.meshgrid(np.arange(D, dtype=np.float32),
                          np.arange(H, dtype=np.float32),
                          np.arange(W, dtype=np.float32),
                          indexing='ij')
    coords = (z + dz, y + dy, x + dx)
    deformed = map_coordinates(vol, coords, order=order, mode=mode, cval=float(cval))
    return deformed.astype(np.float32)


def elastic_deform_3d(vol,
                      sigma=2.0,
                      j=None,
                      order=3,
                      mode='constant',
                      cval=0.0,
                      seed=None,
                      clip_range=(0.0, 400.0),
                      prefer_elasticdeform=True):
    """
    Apply a 3D elastic deformation analogous to a 2D deform_random_grid.
    - vol: float32 [Z,Y,X], background=0
    - sigma: same meaning as in elasticdeform.deform_random_grid
    - j: if None, random in [4..9]; points = 1 + 4*j
    - order: 3 = cubic B-spline; 1 = faster
    - mode/cval: boundary handling (constant 0 keeps background black)
    - seed: reproducibility
    - clip_range: clamp outputs if desired (set None to skip)
    """
    if j is None:
        j = np.random.randint(4, 10)
    points = 1 + 4*j  # 17..37

    if prefer_elasticdeform and _HAS_ED:
        vol_def = _ed.deform_random_grid(
            vol, sigma=sigma, points=points,
            order=order, mode=mode, cval=float(cval), axis=(0, 1, 2),
        ).astype(np.float32)
    else:
        max_dim = float(max(vol.shape))
        spacing = max_dim / float(points - 1)      # ~ 11..25 for 400^3
        sigma_grid = 0.5 * spacing                 # smoothness of the field
        alpha = 0.8 * spacing                      # warp amplitude (voxels)
        vol_def = _elastic_deform_3d_scipy(
            vol, sigma_grid=sigma_grid, alpha=alpha,
            order=order, mode=mode, cval=cval, seed=seed
        )

    if clip_range is not None:
        vmin, vmax = clip_range
        np.clip(vol_def, vmin, vmax, out=vol_def)
    return vol_def


# -----------------------
# Attenuation map
# -----------------------
def make_attenuation_map(vol, inside_value=0.095, threshold=1e-6):
    """
    vol: float32 [Z,Y,X], background=0, objects>0
    returns a float32 map where voxels > threshold are set to inside_value, else 0.
    """
    att = np.zeros_like(vol, dtype=np.float32)
    att[vol > threshold] = float(inside_value)
    return att


# -----------------------
# Raw .img writer (Interfile)
# -----------------------
def save_img_raw_float32(vol, filepath, order='xyz'):
    """
    Save volume as contiguous float32 raw .img.
    - vol is [Z,Y,X] in memory (NumPy default indexing).
    - order='xyz' writes data X-fastest (transpose to [X,Y,Z] before tofile()).
      This is what most Interfile readers expect when matrix size [1]=X etc.
    - order='zyx' writes the array as-is.
    """
    arr = vol.astype(np.float32, copy=False)
    if order == 'xyz':
        arr = np.transpose(arr, (2, 1, 0))  # [Z,Y,X] -> [X,Y,Z] contiguous (X fastest)
    elif order == 'zyx':
        pass  # write as-is
    else:
        raise ValueError("order must be 'xyz' or 'zyx'")
    with open(filepath, 'wb') as f:
        arr.tofile(f)


def write_interfile_header_3d(hdr_path, img_filename, shape_zyx, voxel_size_mm=(2.0, 2.0, 2.0)):
    """
    Interfile 3D header minimal & clean for float32 raw.
    We declare matrix size [1]=X, [2]=Y, [3]=Z to match 'xyz' save order.
    """
    Dz, Dy, Dx = map(int, shape_zyx)          # NumPy shape order
    sx, sy, sz = map(float, voxel_size_mm)    # spacing along X,Y,Z

    with open(hdr_path, 'w') as f:
        f.write("!INTERFILE  :=\n")
        f.write(f"name of data file := {img_filename}\n")
        f.write("!GENERAL DATA :=\n")
        f.write("!GENERAL IMAGE DATA :=\n")
        f.write("!type of data := PET\n")
        f.write("imagedata byte order := LITTLEENDIAN\n")
        f.write("!PET STUDY (General) :=\n")
        f.write("!originating system := simpet3d\n")
        f.write("!STATIC STUDY (General) :=\n")
        f.write("number of dimensions := 3\n")
        # sizes in X,Y,Z (not NumPy order)
        f.write(f"!matrix size [1] := {Dx}\n")  # X
        f.write(f"!matrix size [2] := {Dy}\n")  # Y
        f.write(f"!matrix size [3] := {Dz}\n")  # Z
        f.write("!number format := short float\n")      # float32
        f.write("!number of bytes per pixel := 4\n")    # float32
        f.write(f"scaling factor (mm/pixel) [1] := {sx}\n")
        f.write(f"scaling factor (mm/pixel) [2] := {sy}\n")
        f.write(f"scaling factor (mm/pixel) [3] := {sz}\n")
        f.write("first pixel offset (mm) [1] := 0\n")
        f.write("first pixel offset (mm) [2] := 0\n")
        f.write("first pixel offset (mm) [3] := 0\n")
        f.write("data rescale offset := 0\n")
        f.write("data rescale slope := 1\n")
        f.write("quantification units := 1\n")
        f.write("!image duration (sec) := 1\n")
        f.write("!image start time (sec) := 0\n")
        f.write("!END OF INTERFILE :=\n")


def save_interfile_pair_3d(vol, out_basename,
                           voxel_size_mm=(2.0, 2.0, 2.0),
                           order='xyz'):
    """
    Saves out_basename.img (float32) + out_basename.hdr (Interfile 3D).
    - vol is NumPy float32 array [Z,Y,X].
    - order='xyz' writes X-fastest, and header sizes are [1]=X,[2]=Y,[3]=Z.
    """
    img_path = out_basename + ".img"
    hdr_path = out_basename + ".hdr"
    save_img_raw_float32(vol, img_path, order=order)
    write_interfile_header_3d(
        hdr_path,
        os.path.basename(img_path),
        vol.shape,
        voxel_size_mm
    )
    return img_path, hdr_path


# -----------------------
# Small helpers for names
# -----------------------
def info_sinogramme(diam, r_scatter, r_random):
    """
    Helper to generate slightly randomized scatter/random fractions and prompt counts.
    (In this script 'diam' is not used directly, but kept for compatibility.)
    """
    r_scatter = round(random.uniform(0.95*r_scatter, 1.05*r_scatter), 3)
    r_random = round(random.uniform(0.95*r_random, 1.05*r_random), 3)
    prompt = random.randint(2*10**5, 10**7)
    s_frac = format(r_scatter*0.01, ".2f")  # fractions (0..1) avec 2 décimales
    r_frac = format(r_random*0.01, ".2f")
    return s_frac, r_frac, prompt  # str, str, int


def _sanitize_num_str(s: str) -> str:
    return s.replace('.', 'p')


def build_basename(prefix, out_dir, s_frac, r_frac, prompt, uid, extra_suffix=""):
    s_safe = _sanitize_num_str(s_frac)
    r_safe = _sanitize_num_str(r_frac)
    name = f"{prefix}_s{s_safe}_r{r_safe}_p{prompt}_{uid}{extra_suffix}"
    return os.path.join(out_dir, name)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -----------------------
# High-level generator
# -----------------------
def generate_simpet_sample(
    out_dir="./SIMPET_samples",
    shape=(400, 400, 400),
    voxel_size_mm=(2.0, 2.0, 2.0),
    n_structures_range=(10, 50),
    attenuation_inside_value=0.095,
    seed=None,
):
    """
    High-level function:
      - creates a unique sample folder inside out_dir
      - generates emission volume (phantom + elastic deformation)
      - creates attenuation map
      - saves .npy + Interfile pairs
      - returns a dict with paths + volumes

    Parameters
    ----------
    out_dir : str
        Base directory where a subfolder 'sampleXXXXXX' will be created.
    shape : tuple of int
        (Z,Y,X) volume size.
    voxel_size_mm : tuple of float
        (sx,sy,sz) voxel spacing in mm.
    n_structures_range : (int, int)
        Min/max number of inner structures.
    attenuation_inside_value : float
        Value assigned inside objects in the attenuation map (cm^-1 typically).
    seed : int or None
        Global seed for reproducibility (if None, randomness is used).
    """
    ensure_dir(out_dir)

    # 1) sinogram-like info for filenames (kept for compatibility / realism)
    s_frac, r_frac, prompt = info_sinogramme(
        diam=300, r_scatter=35, r_random=40
    )

    # 2) uid + dedicated folder
    rng = np.random.default_rng(seed)
    uid = int(rng.integers(1, 1_000_000))
    sample_dir = os.path.join(out_dir, f"sample{uid}")
    ensure_dir(sample_dir)

    # 3) base names inside this folder
    em_base = build_basename(
        "SIMPET_emission", sample_dir, s_frac, r_frac, prompt, uid,
        extra_suffix=f"_{int(voxel_size_mm[0])}mmvox"
    )
    att_base = build_basename(
        "SIMPET_attenuation", sample_dir, s_frac, r_frac, prompt, uid,
        extra_suffix=f"_{int(voxel_size_mm[0])}mmvox"
    )

    # 4) generate phantom
    vol, meta = generate_phantom3d(
        shape=shape,
        center_Amax=int(shape[0] * 0.4),   # 160 for 400^3, proportional otherwise
        center_ratio_range=(0.6, 0.8),
        center_intensity=20.0,
        n_structures_range=n_structures_range,
        size_range=(5, 30),
        ellipsoid_prob=0.6,
        child_intensity_range=(5.0, 400.0),
        parent_rot_range_deg=(0.0, 180.0),
        child_rot_range_deg=(0.0, 180.0),
        seed=seed,
    )

    # 5) elastic deformation
    j = np.random.randint(5, 10)
    vol_def = elastic_deform_3d(
        vol,
        sigma=2.0,
        j=j,
        order=3,
        mode='constant',
        cval=0.0,
        seed=None if seed is None else seed + 1,
        clip_range=(0.0, 400.0),
        prefer_elasticdeform=True
    )

    # 6) attenuation map
    att = make_attenuation_map(
        vol_def, inside_value=attenuation_inside_value, threshold=1e-6
    )

    # 7) save numpy arrays
    emission_npy = os.path.join(sample_dir, "emission.npy")
    attenuation_npy = os.path.join(sample_dir, "attenuation.npy")
    np.save(emission_npy, vol_def.astype(np.float32))
    np.save(attenuation_npy, att.astype(np.float32))

    # 8) save Interfile pairs
    em_img, em_hdr = save_interfile_pair_3d(
        vol_def, em_base, voxel_size_mm=voxel_size_mm, order='xyz'
    )
    att_img, att_hdr = save_interfile_pair_3d(
        att, att_base, voxel_size_mm=voxel_size_mm, order='xyz'
    )

    result = dict(
        out_dir=sample_dir,
        emission_volume=vol_def,
        attenuation_volume=att,
        emission_npy=emission_npy,
        attenuation_npy=attenuation_npy,
        emission_img=em_img,
        emission_hdr=em_hdr,
        attenuation_img=att_img,
        attenuation_hdr=att_hdr,
        meta=meta,
        s_frac=s_frac,
        r_frac=r_frac,
        prompt=prompt,
        uid=uid,
    )
    return result


# -----------------------
# CLI
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a 3D SIMPET phantom (emission + attenuation)."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./SIMPET_samples",
        help="Base output directory. A subfolder 'sampleXXXXXX' will be created."
    )
    parser.add_argument(
        "--shape",
        nargs=3,
        type=int,
        default=[400, 400, 400],
        metavar=("Z", "Y", "X"),
        help="Volume shape (Z Y X). Default: 400 400 400"
    )
    parser.add_argument(
        "--voxel_size",
        nargs=3,
        type=float,
        default=[2.0, 2.0, 2.0],
        metavar=("SX", "SY", "SZ"),
        help="Voxel size in mm (SX SY SZ). Default: 2 2 2"
    )
    parser.add_argument(
        "--n_structures_min",
        type=int,
        default=10,
        help="Minimum number of inner structures. Default: 10"
    )
    parser.add_argument(
        "--n_structures_max",
        type=int,
        default=50,
        help="Maximum number of inner structures. Default: 50"
    )
    parser.add_argument(
        "--atten_inside",
        type=float,
        default=0.095,
        help="Attenuation value inside objects (e.g. 0.095 cm^-1). Default: 0.095"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. Default: None (random)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    shape = tuple(args.shape)
    voxel_size = tuple(args.voxel_size)
    n_struct_range = (args.n_structures_min, args.n_structures_max)

    result = generate_simpet_sample(
        out_dir=args.out_dir,
        shape=shape,
        voxel_size_mm=voxel_size,
        n_structures_range=n_struct_range,
        attenuation_inside_value=args.atten_inside,
        seed=args.seed,
    )

    # Single concise print so terminal usage is clean
    print("\nSIMPET 3D sample generated.")
    print(f"  Sample folder      : {result['out_dir']}")
    print(f"  Emission   (.npy)  : {result['emission_npy']}")
    print(f"  Attenuation(.npy)  : {result['attenuation_npy']}")
    print(f"  Emission   (.img)  : {result['emission_img']}")
    print(f"  Emission   (.hdr)  : {result['emission_hdr']}")
    print(f"  Attenuation(.img)  : {result['attenuation_img']}")
    print(f"  Attenuation(.hdr)  : {result['attenuation_hdr']}\n")


if __name__ == "__main__":
    main()
