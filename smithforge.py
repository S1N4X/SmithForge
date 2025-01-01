#!/usr/bin/env python3

#⠀⠀⠀⠀⠀⠀⢰⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⣶⡄⠀⠀⠀⠀⠀
#⠀⠹⣿⣿⣿⣿⡇⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⢠⣄⡀⠀⠀
#⠀⠀⠙⢿⣿⣿⡇⢸⣿⣿⣿ SMITHFORGE ⣿⣿⣿⣿⢸⣿⣿⡶⠀
#⠀⠀⠀⠀⠉⠛⠇⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠸⠟⠋⠀⠀
#⠀⠀⠀⠀⠀⠀⠀⠀⠸⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠿⠇⠀⠀⠀⠀⠀
#⠀⠀ ⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣶⣶⣶⣶⣶⣶⣶⣶⡀⠀⠀⠀⠀⠀⠀⠀⠀
#⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⣿⣿⣿ by ⣿⣿⣿⣷⡀⠀⠀⠀⠀⠀⠀⠀⠀
# ⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣿⣿  S1N4X  ⣿⣿⣿⣄⠀⠀⠀⠀⠀⠀⠀
#⠀⠀⠀⠀⠀⠀⣀⣀⣈⣉⣉⣉⣉⣉⣉⣉⣉⣉⣉⣉⣉⣉⣉⣁⣀⣀⠀⠀⠀⠀
#⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠀⠀⠀⠀
#
# GPL-3.0-only License

import trimesh
from trimesh.exchange import load
from trimesh import transformations as tf
import shapely.geometry
import argparse

def extract_main_mesh(scene):
    if isinstance(scene, trimesh.Scene):
        return trimesh.util.concatenate(scene.dump())
    elif isinstance(scene, trimesh.Trimesh):
        return scene
    else:
        raise ValueError("Unsupported 3MF content.")

def ensure_watertight(mesh):
    if not mesh.is_watertight:
        print("Mesh is not watertight. Attempting to fix...")
        mesh.fill_holes()
        if not mesh.is_watertight:
            raise ValueError("Mesh could not be made watertight.")

def modify_3mf(hueforge_path, base_path, output_path,
               scaledown, rotate_base,
               xshift, yshift, zshift,
               force_scale=None, fill=None, watertight=False):
    """
    1) Rotate the base around Z by --rotatebase degrees (if nonzero).
    2) Compute scale so Hueforge fully occupies at least one dimension => scale = max(scale_x, scale_y).
    3) If scale < 1 and not --scaledown, clamp scale to 1.
    4) Center Hueforge on the base in (x, y).
    5) Embed Hueforge in Z for real overlap (0.1 mm by default).
    6) Apply user-specified shifts: --xshift, --yshift, --zshift
    7) Build a 2D convex hull from base's XY, extrude => 'cutter'.
    8) Intersect Hueforge with that cutter => clip outside base shape.
    9) [NEW STEP] Compute leftover patch from base - Hueforge, extrude, and union.
    10) Union clipped Hueforge + base => single manifold => export.
    11) Optionally ensure the final mesh is watertight.
    """

    print(f"Loading Hueforge: {hueforge_path}")
    hueforge_scene = load.load(hueforge_path)
    hueforge = extract_main_mesh(hueforge_scene)

    print(f"Loading base: {base_path}")
    base_scene = load.load(base_path)
    base = extract_main_mesh(base_scene)

    # ----------------------
    # STEP 1) Rotate the base if requested
    # ----------------------
    if rotate_base != 0:
        print(f"Rotating base by {rotate_base} degrees around Z-axis.")
        angle_radians = rotate_base * 3.14159265359 / 180.0
        rotation_matrix = tf.rotation_matrix(angle_radians, [0, 0, 1])
        base.apply_transform(rotation_matrix)

    # ----------------------
    # STEP 2) Scale Hueforge => fill at least one dimension
    # ----------------------
    hf_min, hf_max = hueforge.bounds
    base_min, base_max = base.bounds

    hueforge_width  = hf_max[0] - hf_min[0]
    hueforge_height = hf_max[1] - hf_min[1]
    base_width      = base_max[0] - base_min[0]
    base_height     = base_max[1] - base_min[1]

    if force_scale is not None:
        uniform_scale = force_scale
        print(f"Using forced scale value: {uniform_scale}")
    else:
        scale_x = base_width  / hueforge_width
        scale_y = base_height / hueforge_height
        uniform_scale = max(scale_x, scale_y)

        if uniform_scale < 1.0 and not scaledown:
            print(f"Computed scale={uniform_scale:.3f} < 1.0, clamping to 1.0 (default).")
            uniform_scale = 1.0

    print("=== Scale Hueforge ===")
    print(f" - Hueforge original dims:  W={hueforge_width:.2f}, H={hueforge_height:.2f}")
    print(f" - Base dims:               W={base_width:.2f},  H={base_height:.2f}")
    if force_scale is None:
        print(f" - scale_x={scale_x:.3f}, scale_y={scale_y:.3f}")
    print(f" - final uniform_scale={uniform_scale:.3f}")

    hueforge.apply_scale([uniform_scale, uniform_scale, 1.0])
    hf_min, hf_max = hueforge.bounds

    # ----------------------
    # STEP 3) Center Hueforge on base in X,Y
    # ----------------------
    base_center_x = (base_min[0] + base_max[0]) / 2.0
    base_center_y = (base_min[1] + base_max[1]) / 2.0

    hf_center_x = (hf_min[0] + hf_max[0]) / 2.0
    hf_center_y = (hf_min[1] + hf_max[1]) / 2.0

    shift_x = base_center_x - hf_center_x
    shift_y = base_center_y - hf_center_y
    hueforge.apply_translation([shift_x, shift_y, 0])
    print(f"Center Hueforge => shift=({shift_x:.2f}, {shift_y:.2f})")

    # ----------------------
    # STEP 4) Embed Hueforge in Z
    # ----------------------
    hf_min, hf_max = hueforge.bounds
    base_top_z = base_max[2]
    hueforge_bottom_z = hf_min[2]

    # Align bottom of Hueforge to top of base
    hueforge.apply_translation([0, 0, base_top_z - hueforge_bottom_z])
    overlap_amount = 0.1
    hueforge.apply_translation([0, 0, -overlap_amount])
    print(f"Embedding Hueforge by {overlap_amount} mm into base for overlap.")

    # ----------------------
    # STEP 5) Apply user-specified shifts
    # ----------------------
    if (xshift != 0) or (yshift != 0) or (zshift != 0):
        print(f"Applying user shifts => X={xshift}, Y={yshift}, Z={zshift}")
        hueforge.apply_translation([xshift, yshift, zshift])

    # ----------------------
    # STEP 6) Build 2D convex hull => extrude
    # ----------------------
    base_verts_2d = [(v[0], v[1]) for v in base.vertices]
    hull_2d = shapely.geometry.MultiPoint(base_verts_2d).convex_hull
    if hull_2d.is_empty:
        print("❌ Base hull is empty—check your base geometry.")
        return

    extrude_height = 500.0
    cutter = trimesh.creation.extrude_polygon(hull_2d, height=extrude_height)

    # ----------------------
    # STEP 7) Intersect => clip Hueforge outside base shape
    # ----------------------
    print("Clipping Hueforge with extruded base hull (intersection)...")
    hueforge_clipped = hueforge.intersection(cutter)
    if hueforge_clipped.is_empty:
        print("❌ Intersection is empty. Possibly no overlap or base not a volume.")
        return

    # ======================================================
    # STEP 8) Create leftover patch to fill uncovered region
    # ------------------------------------------------------
    # We'll do a 2D difference: base_2d - hueforge_2d
    # then extrude that shape in Z so it's flush with base's top surface.
    # If nothing is leftover, we skip it.
    if fill:
        print("Computing leftover patch (base minus Hueforge in XY)...")
        base_pts_2d = [(v[0], v[1]) for v in base.vertices]
        hue_pts_2d  = [(v[0], v[1]) for v in hueforge.vertices]

        base_2d_poly = shapely.geometry.MultiPoint(base_pts_2d).convex_hull
        hue_2d_poly  = shapely.geometry.MultiPoint(hue_pts_2d).convex_hull

        leftover_2d = base_2d_poly.difference(hue_2d_poly)
        if leftover_2d.is_empty:
            print("No leftover area => nothing to fill.")
            patch_3d = None
        else:
            leftover_thickness = fill if fill else 1.0  # default 1mm if no --fill
            print(f"Extruding leftover patch by {leftover_thickness} mm.")
            patch_3d = trimesh.creation.extrude_polygon(leftover_2d, height=leftover_thickness)

            # Place the patch at the top of the base
            patch_3d.apply_translation([0, 0, base_top_z])

    # ----------------------
    # STEP 9) Union clipped Hueforge + base => final mesh
    # ----------------------
    print("Union clipped Hueforge + base => final mesh...")
    if patch_3d:
        # First union base + leftover patch => then union with Hueforge
        base_plus_patch = base.union(patch_3d)
        final_mesh = base_plus_patch.union(hueforge_clipped)
    else:
        # Nothing leftover to fill
        final_mesh = base.union(hueforge_clipped)

    # ----------------------
    # STEP 10) Ensure final mesh is watertight (optional)
    # ----------------------
    if watertight:
        print("Ensuring final mesh is watertight...")
        ensure_watertight(final_mesh)

    # ----------------------
    # STEP 11) Export
    # ----------------------
    print(f"Exporting final mesh to {output_path}")
    final_mesh.export(output_path)
    print("✅ Done! Rotation, user shift, scaling, centering, leftover patch, clipping, embedding, union, and watertight check complete.")


# ----------------------
# MAIN
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine two 3MF models by overlaying one (Hueforge) onto another (base) with automatic scaling, positioning, leftover patch fill, etc."
    )

    # File paths
    parser.add_argument("-f", "--hueforge", required=True, help="Path to Hueforge 3MF file")
    parser.add_argument("-b", "--base", required=True, help="Path to base 3MF file")
    parser.add_argument("-o", "--output", default="combined.3mf", help="Output 3MF file path")

    # Geometry
    parser.add_argument("--rotatebase", type=int, default=0,
                        help="Rotate the base by these many degrees around Z. Example: 90, 180, 270.")

    parser.add_argument("-s", "--scale", type=float,
                        help="Force a specific scale value instead of auto-computing. Examples: 0.5 (scale down by half), 1.0 (no scaling), 2.0 (double size)")

    parser.add_argument("--scaledown", action="store_true",
                        help="If set, allow scale < 1.0. Otherwise, clamp scale to 1.0 if computed scale < 1.0.")

    parser.add_argument("-x","--xshift", type=float, default=0.0, help="Shift hueforge in X before embedding it on the base (mm)")
    parser.add_argument("-y","--yshift", type=float, default=0.0, help="Shift hueforge in Y before embedding it on the base (mm)")
    parser.add_argument("-z","--zshift", type=float, default=0.0, help="Shift hueforge in Z before embedding it on the base (mm)")

    parser.add_argument("--watertight", action="store_true",
                        help="Ensure the final mesh is watertight by filling holes.")

    # [NEW] Optional leftover patch thickness if user wants finer or thicker fill
    # NOT WORKING: Maybe try and embedding 0.1 for the patch
    parser.add_argument("--fill", type=float,
                        help="Thickness (in mm) for leftover patch. If omitted, defaults to 1.0 mm.")

    args = parser.parse_args()
    modify_3mf(
        hueforge_path=args.hueforge,
        base_path=args.base,
        output_path=args.output,
        scaledown=args.scaledown,
        rotate_base=args.rotatebase,
        xshift=args.xshift,
        yshift=args.yshift,
        zshift=args.zshift,
        force_scale=args.scale,
        fill=args.fill,
        watertight=args.watertight
    )