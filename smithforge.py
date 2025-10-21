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
import zipfile
import xml.etree.ElementTree as ET
import json
import os
import tempfile
import shutil

# Import repair module
try:
    from repair import auto_repair_mesh, RepairReport
except ImportError:
    # If running from parent directory
    try:
        from smithforge.repair import auto_repair_mesh, RepairReport
    except ImportError:
        print("⚠️  Warning: Could not import repair module. Auto-repair will be disabled.")
        auto_repair_mesh = None
        RepairReport = None

# Configuration constants
DEFAULT_EMBEDDING_OVERLAP_MM = 0.1  # Default Z-axis overlap for proper model union

def extract_main_mesh(scene):
    if isinstance(scene, trimesh.Scene):
        return trimesh.util.concatenate(scene.dump())
    elif isinstance(scene, trimesh.Trimesh):
        return scene
    else:
        raise ValueError("Unsupported 3MF content.")

def extract_color_layers(hueforge_3mf_path):
    """
    Extract color layer information from a Hueforge 3MF file.

    Returns:
        dict with 'layers' (list of dicts with top_z, extruder, color),
        'filament_colours' (list of hex color strings), and optionally
        'layer_config_ranges_xml' (raw XML string), or None if no color data found.
    """
    try:
        with zipfile.ZipFile(hueforge_3mf_path, 'r') as zf:
            # Extract color layers from custom_gcode_per_layer.xml
            layers = []
            try:
                with zf.open('Metadata/custom_gcode_per_layer.xml') as f:
                    tree = ET.parse(f)
                    root = tree.getroot()

                    # Find all layer elements
                    for layer_elem in root.findall('.//layer[@type="2"]'):
                        top_z = layer_elem.get('top_z')
                        extruder = layer_elem.get('extruder')
                        color = layer_elem.get('color')

                        if top_z and extruder and color:
                            layers.append({
                                'top_z': float(top_z),
                                'extruder': extruder,
                                'color': color
                            })
            except KeyError:
                print("ℹ️  No custom_gcode_per_layer.xml found in Hueforge 3MF")
                return None

            # Extract filament colors from project_settings.config
            filament_colours = []
            try:
                with zf.open('Metadata/project_settings.config') as f:
                    config_str = f.read().decode('utf-8')
                    # Parse as JSON (it's a JSON file)
                    config = json.loads(config_str)
                    filament_colours = config.get('filament_colour', [])
            except (KeyError, json.JSONDecodeError) as e:
                print(f"ℹ️  Could not extract filament colors: {e}")

            # Extract layer_config_ranges.xml if it exists
            layer_config_ranges_xml = None
            try:
                with zf.open('Metadata/layer_config_ranges.xml') as f:
                    layer_config_ranges_xml = f.read().decode('utf-8')
                    print("✅ Extracted layer_config_ranges.xml")
            except KeyError:
                print("ℹ️  No layer_config_ranges.xml found in Hueforge 3MF")

            if layers:
                print(f"✅ Extracted {len(layers)} color layer transitions")
                result = {
                    'layers': layers,
                    'filament_colours': filament_colours
                }
                if layer_config_ranges_xml:
                    result['layer_config_ranges_xml'] = layer_config_ranges_xml
                return result
            else:
                return None

    except Exception as e:
        print(f"⚠️  Error extracting color layers: {e}")
        return None

def inject_color_metadata(output_3mf_path, color_data, z_offset):
    """
    Inject color layer metadata into an exported 3MF file.

    Args:
        output_3mf_path: Path to the 3MF file to modify
        color_data: Dict with 'layers' and 'filament_colours'
        z_offset: Z offset to add to all layer heights
    """
    try:
        # Create a temporary directory to work with the 3MF contents
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract the 3MF
            with zipfile.ZipFile(output_3mf_path, 'r') as zf:
                zf.extractall(temp_dir)

            # Create Metadata directory if it doesn't exist
            metadata_dir = os.path.join(temp_dir, 'Metadata')
            os.makedirs(metadata_dir, exist_ok=True)

            # Create custom_gcode_per_layer.xml with adjusted Z heights
            custom_gcode_xml = ET.Element('custom_gcodes_per_layer')
            plate = ET.SubElement(custom_gcode_xml, 'plate')
            ET.SubElement(plate, 'plate_info', id='1')

            for layer_info in color_data['layers']:
                adjusted_z = layer_info['top_z'] + z_offset
                ET.SubElement(plate, 'layer',
                            top_z=f"{adjusted_z:.17g}",
                            type='2',
                            extruder=layer_info['extruder'],
                            color=layer_info['color'],
                            extra='',
                            gcode='tool_change')

            ET.SubElement(plate, 'mode', value='MultiAsSingle')

            # Write the XML file
            tree = ET.ElementTree(custom_gcode_xml)
            ET.indent(tree, space='')
            custom_gcode_path = os.path.join(metadata_dir, 'custom_gcode_per_layer.xml')
            tree.write(custom_gcode_path, encoding='utf-8', xml_declaration=True)

            # Update or create project_settings.config with filament colors
            project_settings_path = os.path.join(metadata_dir, 'project_settings.config')
            if os.path.exists(project_settings_path):
                # Read existing config and update
                with open(project_settings_path, 'r') as f:
                    config = json.load(f)
            else:
                # Create minimal config
                config = {}

            # Inject filament colors
            if color_data['filament_colours']:
                config['filament_colour'] = color_data['filament_colours']
                config['filament_type'] = ['PLA'] * len(color_data['filament_colours'])

            # Write back
            with open(project_settings_path, 'w') as f:
                json.dump(config, f, indent=4)

            # Create Bambu Lab identification files to prevent "not from Bambu Lab" warning
            # This ensures Bambu Studio recognizes the file and preserves color metadata

            # Create slice_info.config with Bambu Lab identifiers
            slice_info_xml = ET.Element('config')
            header = ET.SubElement(slice_info_xml, 'header')
            ET.SubElement(header, 'header_item', key='X-BBL-Client-Type', value='slicer')
            ET.SubElement(header, 'header_item', key='X-BBL-Client-Version', value='02.00.03.54')

            slice_info_tree = ET.ElementTree(slice_info_xml)
            ET.indent(slice_info_tree, space='  ')
            slice_info_path = os.path.join(metadata_dir, 'slice_info.config')
            slice_info_tree.write(slice_info_path, encoding='UTF-8', xml_declaration=True)

            # Create minimal model_settings.config for Bambu Studio compatibility
            model_settings_xml = ET.Element('config')
            obj = ET.SubElement(model_settings_xml, 'object', id='1')
            ET.SubElement(obj, 'metadata', key='name', value='SmithForge Combined Model')
            ET.SubElement(obj, 'metadata', key='extruder', value='1')

            part = ET.SubElement(obj, 'part', id='1', subtype='normal_part')
            ET.SubElement(part, 'metadata', key='name', value='Combined')
            ET.SubElement(part, 'metadata', key='matrix', value='1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1')

            plate_elem = ET.SubElement(model_settings_xml, 'plate')
            ET.SubElement(plate_elem, 'metadata', key='plater_id', value='1')
            ET.SubElement(plate_elem, 'metadata', key='plater_name', value='')
            ET.SubElement(plate_elem, 'metadata', key='locked', value='false')

            # Set up filament mapping based on number of colors
            num_filaments = len(color_data.get('filament_colours', []))
            if num_filaments > 0:
                filament_maps = ' '.join([str(i) for i in range(1, num_filaments + 1)])
                ET.SubElement(plate_elem, 'metadata', key='filament_maps', value=filament_maps)

            model_instance = ET.SubElement(plate_elem, 'model_instance')
            ET.SubElement(model_instance, 'metadata', key='object_id', value='1')
            ET.SubElement(model_instance, 'metadata', key='instance_id', value='0')

            assemble = ET.SubElement(model_settings_xml, 'assemble')
            ET.SubElement(assemble, 'assemble_item',
                         object_id='1',
                         instance_id='0',
                         transform='1 0 0 0 1 0 0 0 1 0 0 0',
                         offset='0 0 0')

            model_settings_tree = ET.ElementTree(model_settings_xml)
            ET.indent(model_settings_tree, space='  ')
            model_settings_path = os.path.join(metadata_dir, 'model_settings.config')
            model_settings_tree.write(model_settings_path, encoding='UTF-8', xml_declaration=True)

            print("✅ Added Bambu Lab identification metadata for compatibility")

            # Copy layer_config_ranges.xml if it exists and adjust Z values
            if 'layer_config_ranges_xml' in color_data:
                try:
                    # Parse the XML
                    ranges_root = ET.fromstring(color_data['layer_config_ranges_xml'])

                    # Adjust all min_z and max_z values
                    for range_elem in ranges_root.findall('.//range'):
                        min_z = range_elem.get('min_z')
                        max_z = range_elem.get('max_z')

                        if min_z:
                            adjusted_min = float(min_z) + z_offset
                            range_elem.set('min_z', str(adjusted_min))

                        if max_z:
                            adjusted_max = float(max_z) + z_offset
                            range_elem.set('max_z', str(adjusted_max))

                    # Write the adjusted XML
                    ranges_tree = ET.ElementTree(ranges_root)
                    ET.indent(ranges_tree, space=' ')
                    ranges_path = os.path.join(metadata_dir, 'layer_config_ranges.xml')
                    ranges_tree.write(ranges_path, encoding='utf-8', xml_declaration=True)
                    print("✅ Added layer_config_ranges.xml with adjusted Z values")
                except Exception as e:
                    print(f"⚠️  Could not process layer_config_ranges.xml: {e}")

            # Repack the 3MF
            temp_output = output_3mf_path + '.tmp'
            with zipfile.ZipFile(temp_output, 'w', zipfile.ZIP_DEFLATED) as zf_out:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zf_out.write(file_path, arcname)

            # Replace original with modified
            shutil.move(temp_output, output_3mf_path)
            print(f"✅ Injected color metadata with Z-offset {z_offset:.3f} mm")

    except Exception as e:
        print(f"⚠️  Error injecting color metadata: {e}")

def sample_perimeter_height(mesh, num_samples=40):
    """
    Sample Z-heights along the perimeter of a mesh to detect the background height.

    Uses the mesh boundary to sample points and finds the most common height value,
    which typically represents the Hueforge background layer.

    Args:
        mesh: trimesh.Trimesh object
        num_samples: Number of points to sample along the perimeter (default 40)

    Returns:
        float: The most common (mode) Z-height value from perimeter samples
    """
    import numpy as np

    # Get the 2D boundary of the mesh (edges that appear only once)
    # We'll work in XY plane projection
    outline = mesh.outline()

    if outline is None or len(outline.entities) == 0:
        print("⚠️  Warning: Could not extract mesh outline, using mesh bounds")
        # Fallback: use the minimum Z value
        return mesh.bounds[0][2]

    # Sample points along the outline
    z_heights = []

    for entity in outline.entities:
        # Get points from this path entity
        points = entity.discrete(outline.vertices)

        # For each 2D point, find closest vertex in original mesh and get its Z
        for point_2d in points:
            # Find vertices close to this XY position
            distances = np.sqrt(
                (mesh.vertices[:, 0] - point_2d[0])**2 +
                (mesh.vertices[:, 1] - point_2d[1])**2
            )
            closest_idx = np.argmin(distances)
            z_heights.append(mesh.vertices[closest_idx, 2])

    if len(z_heights) == 0:
        print("⚠️  Warning: No perimeter points sampled, using minimum Z")
        return mesh.bounds[0][2]

    # Limit to num_samples evenly spaced points
    if len(z_heights) > num_samples:
        indices = np.linspace(0, len(z_heights) - 1, num_samples, dtype=int)
        z_heights = [z_heights[i] for i in indices]

    z_heights = np.array(z_heights)

    # Find the mode using histogram binning
    # Use 20 bins to group similar heights
    hist, bins = np.histogram(z_heights, bins=20)
    mode_bin_idx = np.argmax(hist)

    # Return the center of the most common bin
    background_height = (bins[mode_bin_idx] + bins[mode_bin_idx + 1]) / 2.0

    print(f"📏 Sampled {len(z_heights)} perimeter points")
    print(f"📏 Detected background height: {background_height:.3f} mm")
    print(f"📏 Height range: {z_heights.min():.3f} to {z_heights.max():.3f} mm")

    return background_height

def create_fill_geometry(base_mesh, hueforge_mesh, fill_height, base_top_z):
    """
    Create fill geometry to fill gaps between a scaled-down Hueforge overlay and base boundaries.

    Args:
        base_mesh: The base mesh (defines outer boundary)
        hueforge_mesh: The Hueforge overlay mesh (defines inner boundary)
        fill_height: The Z-height at which to create the fill (typically Hueforge background height)
        base_top_z: The Z coordinate of the top of the base mesh

    Returns:
        trimesh.Trimesh: Fill mesh, or None if no gap exists
    """
    import numpy as np

    # Get 2D projections (XY plane)
    base_verts_2d = [(v[0], v[1]) for v in base_mesh.vertices]
    hf_verts_2d = [(v[0], v[1]) for v in hueforge_mesh.vertices]

    # Create convex hulls for both shapes
    base_hull = shapely.geometry.MultiPoint(base_verts_2d).convex_hull
    hf_hull = shapely.geometry.MultiPoint(hf_verts_2d).convex_hull

    # Check if there's actually a gap to fill
    if hf_hull.contains(base_hull) or hf_hull.equals(base_hull):
        print("ℹ️  Hueforge covers entire base area - no gap filling needed")
        return None

    # Compute the difference region (gap area between base and Hueforge)
    gap_region = base_hull.difference(hf_hull)

    if gap_region.is_empty or gap_region.area < 1e-6:
        print("ℹ️  Gap area is negligible - no fill geometry created")
        return None

    print(f"📐 Gap area detected: {gap_region.area:.2f} mm²")

    # Calculate the height of the fill extrusion
    # Fill should go from base_top_z to the fill_height
    fill_thickness = fill_height - base_top_z

    if fill_thickness <= 0:
        print(f"⚠️  Warning: Fill height ({fill_height:.3f}) is below base top ({base_top_z:.3f})")
        print("    Using minimum thickness of 0.1mm")
        fill_thickness = 0.1

    print(f"📐 Creating fill geometry: thickness = {fill_thickness:.3f} mm")

    # Extrude the gap region to create fill mesh
    try:
        fill_mesh = trimesh.creation.extrude_polygon(gap_region, height=fill_thickness)

        # Position the fill mesh at the correct Z height (on top of base)
        fill_mesh.apply_translation([0, 0, base_top_z])

        print(f"✅ Fill geometry created: {len(fill_mesh.vertices)} vertices, {len(fill_mesh.faces)} faces")

        return fill_mesh

    except Exception as e:
        print(f"⚠️  Failed to create fill geometry: {e}")
        return None

def modify_3mf(hueforge_path, base_path, output_path,
               scaledown, rotate_base,
               xshift, yshift, zshift,
               force_scale=None,
               preserve_colors=False,
               auto_repair=False,
               fill_gaps=False):
    """
    1) Rotate the base around Z by --rotatebase degrees (if nonzero).
    2) Compute scale so Hueforge fully occupies at least one dimension => scale = max(scale_x, scale_y).
    3) If scale < 1 and not --nominimum, clamp scale to 1.
    4) Center Hueforge on the base in (x, y).
    5) Embed Hueforge in Z for real overlap (see DEFAULT_EMBEDDING_OVERLAP_MM constant).
    6) Apply user-specified shifts: --xshift, --yshift, --zshift
    7) Build a 2D convex hull from base's XY, extrude => 'cutter'.
    8) Intersect Hueforge with that cutter => clip outside base shape.
    9) If fill_gaps=True, detect background height and create fill geometry for gaps between overlay and base.
    10) Union clipped Hueforge (+ fill if enabled) + base => single manifold => export.
    11) If preserve_colors=True, extract color layers from Hueforge and inject into output with adjusted Z heights.
    12) If auto_repair=True, automatically validate and repair mesh issues before processing.
    """

    # Extract color layer information if requested
    color_data = None
    if preserve_colors:
        print("🎨 Extracting color layer information from Hueforge...")
        color_data = extract_color_layers(hueforge_path)
        if color_data is None:
            print("ℹ️  No color layer data found, proceeding without color preservation")

    print(f"Loading Hueforge: {hueforge_path}")
    hueforge_scene = load.load(hueforge_path)
    hueforge = extract_main_mesh(hueforge_scene)

    print(f"Loading base: {base_path}")
    base_scene = load.load(base_path)
    base = extract_main_mesh(base_scene)

    # Auto-repair meshes if requested
    if auto_repair and auto_repair_mesh is not None:
        print("\n🔧 === MESH REPAIR MODE ENABLED ===")

        print("\n🔍 Checking Hueforge mesh...")
        hueforge, hueforge_report = auto_repair_mesh(hueforge)
        print(hueforge_report)

        print("\n🔍 Checking base mesh...")
        base, base_report = auto_repair_mesh(base)
        print(base_report)

        print("🔧 === MESH REPAIR COMPLETE ===\n")

        if not hueforge_report.success or not base_report.success:
            print("⚠️  Warning: Some mesh repairs were not fully successful. Boolean operations may still fail.")
    elif auto_repair and auto_repair_mesh is None:
        print("⚠️  Auto-repair requested but repair module not available. Skipping repair.")

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
    overlap_amount = DEFAULT_EMBEDDING_OVERLAP_MM
    hueforge.apply_translation([0, 0, -overlap_amount])
    print(f"Embedding Hueforge by {overlap_amount} mm into base for overlap.")

    # Track Z offset for color layer adjustment (before user shifts)
    z_offset_before_user_shifts = base_top_z - overlap_amount

    # ----------------------
    # STEP 5) Apply user-specified shifts
    # ----------------------
    if (xshift != 0) or (yshift != 0) or (zshift != 0):
        print(f"Applying user shifts => X={xshift}, Y={yshift}, Z={zshift}")
        hueforge.apply_translation([xshift, yshift, zshift])

    # Calculate final Z offset for color layers (including user zshift)
    final_z_offset = z_offset_before_user_shifts + zshift

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

    # ----------------------
    # STEP 7.5) Fill gaps if requested
    # ----------------------
    overlay_to_union = hueforge_clipped
    if fill_gaps:
        print("\n🔧 === GAP FILLING MODE ENABLED ===")

        # Sample perimeter height from the clipped Hueforge
        background_height = sample_perimeter_height(hueforge_clipped)

        # Create fill geometry
        fill_mesh = create_fill_geometry(base, hueforge_clipped, background_height, base_top_z)

        if fill_mesh is not None:
            # Union the fill with the clipped Hueforge
            print("Combining fill geometry with clipped Hueforge...")
            try:
                overlay_to_union = hueforge_clipped.union(fill_mesh)
                print("✅ Fill geometry successfully merged with overlay")
            except Exception as e:
                print(f"⚠️  Failed to merge fill geometry: {e}")
                print("    Proceeding without fill...")
                overlay_to_union = hueforge_clipped

        print("🔧 === GAP FILLING COMPLETE ===\n")

    # ----------------------
    # STEP 8) Union => single manifold
    # ----------------------
    print("Union clipped Hueforge + base => final mesh...")
    final_mesh = base.union(overlay_to_union)

    # ----------------------
    # STEP 9) Export
    # ----------------------
    print(f"Exporting final mesh to {output_path}")
    final_mesh.export(output_path)

    # ----------------------
    # STEP 10) Inject color metadata if requested
    # ----------------------
    if preserve_colors and color_data:
        print(f"🎨 Injecting color layer metadata (Z-offset: {final_z_offset:.3f} mm)...")
        inject_color_metadata(output_path, color_data, final_z_offset)

    print("✅ Done! Rotation, user shift, scaling, centering, clipping, embedding, and union complete.")

# ----------------------
# MAIN
# ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine two 3MF models by overlaying one (Hueforge) onto another (base) with automatic scaling, positioning, and proper model intersection. Optionally rotate base, scale & center Hueforge, allow user shifts, clip to base shape, union."
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
                        help="If set, allow scale < 1.0. Otherwise, clamp scale to 1.0 if computed scale < 1.0. Only used if --scale is not set.")
    
    parser.add_argument("--xshift", type=float, default=0.0, help="Shift hueforge in X before embedding it on the base (mm)")
    parser.add_argument("--yshift", type=float, default=0.0, help="Shift hueforge in Y before embedding it on the base (mm)")
    parser.add_argument("--zshift", type=float, default=0.0, help="Shift hueforge in Z before embedding it on the base (mm)")

    # Color preservation
    parser.add_argument("--preserve-colors", action="store_true",
                        help="Preserve Hueforge color layer information in the output 3MF file (adjusts Z-heights for new position)")

    # Mesh repair
    parser.add_argument("--auto-repair", action="store_true",
                        help="Automatically validate and repair mesh issues before processing (fixes holes, non-manifold edges, degenerate faces, etc.)")

    # Gap filling
    parser.add_argument("--fill-gaps", action="store_true",
                        help="Fill gaps between scaled overlay and base boundaries with background height material (useful when overlay is scaled smaller than base)")

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
        preserve_colors=args.preserve_colors,
        auto_repair=args.auto_repair,
        fill_gaps=args.fill_gaps
    )
