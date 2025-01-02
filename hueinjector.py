#      |________|___________________|_
#      |        | | | | | | | | | | | |________________
#      |________|___________________|_|                ,
#      |        |                   |                  ,
#                                                      hueinjector.py
#                                                      By S1N4X
# GPL-3.0-only License

# TODO:
# 1. Base layer height should be sent via parameter from smithforge.py or with args from the command line. Else, will be set at 0.2mm.
# 2. Layer ranges of the hueforge will be calculated from the user's Hueforge instructions. Required.
# 3. layer_height of the hueforge will be calculated from the user's Hueforge instructions. Required. Else, will be set at 0.08mm for each layers of the hueforge.
# 4. Extruder mapping will be sequential according to coulour mapping in the Hueforge instructions. Else will be set to 1=Black, 2=Silver, 3=Gray, 4=Jade White. 
# 5. Base shape model height will be calculated from the base shape model (coming from smithforge.py) and will be used to calculate the embed_offset. Required.
# 6. Infill will be set to 100% by default for the hueforge. Required. Should be sent via parameter from smithforge.py or with args from the command line
# 7. Infill will be set to 15% by default for the base shape model. Required. Should be sent via parameter from smithforge.py or with args from the command line
# 8. Embed_offset will be calculated from the base shape model height and will be used to embed the hueforge into the base shape model. Required.
# 9. The final 3mf file will be updated with the layer_config_ranges.xml file with the calculated layer ranges and extruder mapping. Required.
# 10. The final 3mf file will be updated with the layer_config.xml file.

# REMOVE COLOR MAP

import os
import zipfile
import xml.etree.ElementTree as ET
import tempfile
import argparse
import re

###############################################################################
# 1) Parse Hueforge Instructions from TXT
###############################################################################
def parse_hueforge_txt(txt_path,
                       default_hueforge_layer_height=0.08,
                       default_colors=None):
    """
    Reads user-provided text file for:
      - Hueforge layer height (fallback=0.08)
      - Base layer height (fallback=0.2)
      - Min/Max Depth
      - Filament color -> extruder mapping
      - Swap instructions
    Returns a dict with keys:
      {
        'base_layer_height': float,
        'hueforge_layer_height': float,
        'min_depth': float,
        'max_depth': float,
        'color_map': { 'black':1, 'silver':2, ...},
        'swaps': list of (z_mm, extruder_index),
        'infill_hueforge': 100,
        'infill_base': 15
      }
    """
    if default_colors is None:
        default_colors = {
            'black': 1,
            'silver': 2,
            'gray': 3,
            'jade white': 4
        }

    # Build a reverse-lower map to handle user input
    color_map_lower = {c.lower(): idx for c, idx in default_colors.items()}

    result = {
        'base_layer_height': 0.2,  # fallback if not in text
        'hueforge_layer_height': default_hueforge_layer_height,
        'min_depth': None,
        'max_depth': None,
        'color_map': dict(color_map_lower),
        'swaps': [],
        'infill_hueforge': 100,
        'infill_base': 15
    }

    if not os.path.isfile(txt_path):
        print(f"⚠️ Hueforge TXT file not found: {txt_path}, using defaults.")
        return result

    with open(txt_path, 'r', encoding='utf-8') as f:
        data = f.read()

    # BASE LAYER: e.g. "Print at 100% infill with a layer height of 0.08mm with a base layer of 0.16mm"
    # We'll attempt a simple regex parse:
    m1 = re.search(r'base layer of\s*([\d.]+)mm', data, re.IGNORECASE)
    if m1:
        result['base_layer_height'] = float(m1.group(1))

    # HUEFORGE LAYER: e.g. "layer height of 0.08mm" (the normal layer for hueforge)
    m2 = re.search(r'layer height of\s*([\d.]+)mm', data, re.IGNORECASE)
    if m2:
        # This might pick up the normal layer height. 
        # But watch out for confusion with base layer. We do a second parse above.
        # We'll only set if it's not the same as base layer:
        val = float(m2.group(1))
        if abs(val - result['base_layer_height']) > 0.0001:
            result['hueforge_layer_height'] = val

    # MIN DEPTH: e.g. "Min Depth of 0.88mm"
    m3 = re.search(r'Min Depth of\s*([\d.]+)mm', data, re.IGNORECASE)
    if m3:
        result['min_depth'] = float(m3.group(1))

    # MAX DEPTH: e.g. "Max Depth is 2.2mm"
    m4 = re.search(r'Max Depth is\s*([\d.]+)mm', data, re.IGNORECASE)
    if m4:
        result['max_depth'] = float(m4.group(1))

    # INFILL for Hueforge, e.g. "Print at 100% infill"
    m5 = re.search(r'Print at\s*(\d+)%\s*infill', data, re.IGNORECASE)
    if m5:
        inf = int(m5.group(1))
        result['infill_hueforge'] = inf

    # Similarly, if the text says "Base shape ... 15% infill" you could parse. 
    # But from your sample, it wasn't explicitly stated, so we keep default=15%.

    # Filament lines + color map. 
    # For demonstration, we assume default color_map. If your text has lines like:
    #  "PLA BambuLab Basic Black Transmission Distance: 0.6"
    # we might parse them. For brevity, let's skip or do minimal:
    lines = data.splitlines()
    extruder_counter = max(result['color_map'].values()) + 1
    for ln in lines:
        # e.g. "PLA BambuLab Basic Red Transmission Distance: 4"
        # We'll pull out "Red" and see if it's known:
        mm = re.search(r'Basic\s+(\S+)\s+Transmission', ln, re.IGNORECASE)
        if mm:
            color_name = mm.group(1).lower()
            if color_name not in result['color_map']:
                result['color_map'][color_name] = extruder_counter
                extruder_counter += 1

    # Swap instructions: 
    # e.g. "At layer #11 (0.96mm) swap to Jade White"
    for ln in lines:
        mswap = re.search(r'At layer #\d+\s*\(([\d.]+)mm\)\s*swap to\s+(.+)', ln, re.IGNORECASE)
        if mswap:
            z_val = float(mswap.group(1))
            col_name = mswap.group(2).strip().lower()
            extruder_idx = result['color_map'].get(col_name, 1)  # default black=1
            result['swaps'].append((z_val, extruder_idx))

    result['swaps'].sort(key=lambda x: x[0])
    return result

###############################################################################
# 2) Insert or create 'layer_config_ranges.xml' and 'layer_config.xml'
###############################################################################
def update_3mf_metadata_ranges(
    three_mf_path,
    base_layer_height,
    hueforge_layer_height,
    min_depth,
    max_depth,
    infill_base,
    infill_hueforge,
    base_model_height,
    hueforge_swaps,
    color_map,
    base_extruder=1,
    start_extruder_hueforge=2
):
    """
    Post-process the .3mf (ZIP) to add:
      1) Metadata/layer_config_ranges.xml  => with <object id="someID"><range>...</range></object>
      2) Metadata/layer_config.xml         => optional global or fallback settings

    We'll treat the final merged object as 'object id=1' for simplicity.
    We'll assume 'min_depth'..'max_depth' apply to Hueforge portion, 
    but we embed Hueforge at 'embed_offset = base_model_height'.

    :param three_mf_path: path to the final 'combined.3mf'
    :param base_layer_height: float (default 0.2)
    :param hueforge_layer_height: float (default 0.08)
    :param min_depth: float
    :param max_depth: float
    :param infill_base: int (%)
    :param infill_hueforge: int (%)
    :param base_model_height: float => used as embed_offset
    :param hueforge_swaps: list of (z_mm, extruder_idx) for hueforge color changes
    :param color_map: { 'black':1, 'silver':2, ... } to fill extruder usage
    :param base_extruder: which extruder prints the base
    :param start_extruder_hueforge: which extruder starts Hueforge
    """
    embed_offset = base_model_height  # Per your specs: Hueforge sits on top of the base

    #############
    # layer_config_ranges.xml
    #############
    ranges_xml_name = "Metadata/layer_config_ranges.xml"
    # Attempt to read existing file or create a fresh <objects/>
    with zipfile.ZipFile(three_mf_path, 'r') as z_in:
        if ranges_xml_name in z_in.namelist():
            original_data = z_in.read(ranges_xml_name)
            root_ranges = ET.fromstring(original_data)
        else:
            root_ranges = ET.Element('objects')  # create if missing

    # We assume final object id=1. 
    object_id = "1"
    # Find or create <object id="1">
    target_obj = None
    for obj in root_ranges.findall('object'):
        if obj.get('id') == object_id:
            target_obj = obj
            break
    if target_obj is None:
        target_obj = ET.SubElement(root_ranges, 'object', attrib={'id': object_id})

    # Clear out old <range> if any:
    for rng in list(target_obj.findall('range')):
        target_obj.remove(rng)

    # 1) Range for the base from z=0..(base_model_height):
    #    We'll do base_layer_height, infill=15 (default). 
    rng_base = ET.SubElement(target_obj, 'range', {
        'min_z': "0.000000",
        'max_z': f"{base_model_height:.6f}"
    })
    ET.SubElement(rng_base, 'option', {'opt_key': 'extruder'}).text = str(base_extruder)
    ET.SubElement(rng_base, 'option', {'opt_key': 'layer_height'}).text = f"{base_layer_height}"
    ET.SubElement(rng_base, 'option', {'opt_key': 'sparse_infill_density'}).text = f"{infill_base}%"

    # 2) Range for the Hueforge from z=base_model_height..(base_model_height + min_depth):
    #    That portion is "the base portion of the Hueforge" => sometimes user might want a "larger" layer 
    #    but in your instructions, you said base layer for Hueforge is also used up to min_depth
    z_hueforge_min = base_model_height
    z_hueforge_base_top = base_model_height + min_depth
    rng_hueforge_base = ET.SubElement(target_obj, 'range', {
        'min_z': f"{z_hueforge_min:.6f}",
        'max_z': f"{z_hueforge_base_top:.6f}"
    })
    ET.SubElement(rng_hueforge_base, 'option', {'opt_key': 'extruder'}).text = str(start_extruder_hueforge)
    ET.SubElement(rng_hueforge_base, 'option', {'opt_key': 'layer_height'}).text = f"{base_layer_height}"
    ET.SubElement(rng_hueforge_base, 'option', {'opt_key': 'sparse_infill_density'}).text = f"{infill_hueforge}%"

    # Next, from z_hueforge_base_top.. up to each swap => hueforge_layer_height
    current_extruder = start_extruder_hueforge
    previous_z = z_hueforge_base_top

    for (swap_z, new_extruder) in hueforge_swaps:
        actual_swap_z = swap_z + embed_offset
        if actual_swap_z > previous_z:
            rng_mid = ET.SubElement(target_obj, 'range', {
                'min_z': f"{previous_z:.6f}",
                'max_z': f"{actual_swap_z:.6f}"
            })
            ET.SubElement(rng_mid, 'option', {'opt_key': 'extruder'}).text = str(current_extruder)
            ET.SubElement(rng_mid, 'option', {'opt_key': 'layer_height'}).text = f"{hueforge_layer_height}"
            ET.SubElement(rng_mid, 'option', {'opt_key': 'sparse_infill_density'}).text = f"{infill_hueforge}%"

        current_extruder = new_extruder
        previous_z = actual_swap_z

    # Finally, from last swap up to (base_model_height + max_depth)
    z_final = base_model_height + max_depth
    if z_final > previous_z:
        rng_top = ET.SubElement(target_obj, 'range', {
            'min_z': f"{previous_z:.6f}",
            'max_z': f"{z_final:.6f}"
        })
        ET.SubElement(rng_top, 'option', {'opt_key': 'extruder'}).text = str(current_extruder)
        ET.SubElement(rng_top, 'option', {'opt_key': 'layer_height'}).text = f"{hueforge_layer_height}"
        ET.SubElement(rng_top, 'option', {'opt_key': 'sparse_infill_density'}).text = f"{infill_hueforge}%"

    # Convert updated <objects> to XML
    final_ranges_data = ET.tostring(root_ranges, encoding='utf-8', method='xml')

    #############
    # layer_config.xml (Global Settings)
    #############
    config_xml_name = "Metadata/layer_config.xml"
    # For demonstration, we’ll store something minimal
    # like <layer_config><base_layer>0.2</base_layer> ...</layer_config>
    root_config = ET.Element('layer_config')

    e_base_lh = ET.SubElement(root_config, 'base_layer_height')
    e_base_lh.text = f"{base_layer_height}"

    e_hue_lh = ET.SubElement(root_config, 'hueforge_layer_height')
    e_hue_lh.text = f"{hueforge_layer_height}"

    e_infill_base = ET.SubElement(root_config, 'infill_base')
    e_infill_base.text = f"{infill_base}%"

    e_infill_hue = ET.SubElement(root_config, 'infill_hueforge')
    e_infill_hue.text = f"{infill_hueforge}%"

    final_config_data = ET.tostring(root_config, encoding='utf-8', method='xml')

    # Rebuild the .3mf
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(three_mf_path, 'r') as z_in:
        z_in.extractall(temp_dir)

    # Overwrite or create 'layer_config_ranges.xml'
    ranges_path = os.path.join(temp_dir, ranges_xml_name)
    os.makedirs(os.path.dirname(ranges_path), exist_ok=True)
    with open(ranges_path, 'wb') as f_out:
        f_out.write(final_ranges_data)

    # Overwrite or create 'layer_config.xml'
    config_path = os.path.join(temp_dir, config_xml_name)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'wb') as f_out:
        f_out.write(final_config_data)

    # Re-zip to finalize
    os.remove(three_mf_path)
    with zipfile.ZipFile(three_mf_path, 'w', zipfile.ZIP_DEFLATED) as z_out:
        for root_dir, dirs, files in os.walk(temp_dir):
            for filename in files:
                abs_path = os.path.join(root_dir, filename)
                rel_path = os.path.relpath(abs_path, temp_dir)
                z_out.write(abs_path, arcname=rel_path)

    print(f"✅ Updated {three_mf_path} with layer_config_ranges.xml and layer_config.xml.")


###############################################################################
# 3) Main or sample usage
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="3MF object layer height modifiers config injection.")
    parser.add_argument("input_3mf",
                        help="Path to the 3MF.")
    parser.add_argument("--hueforge_instructions", default="hueforge.txt",
                        help="Path to the user-provided Hueforge instructions TXT.")
    parser.add_argument("--infill_hueforge", type=int, default=100,
                        help="Default % infill for the Hueforge model if not specified in Hueforge instructions.")
    parser.add_argument("--base_model_height", type=float, default=2.0,
                        help="Height (mm) of the base model to compute embed offset.")
    parser.add_argument("--base_layer_height", type=float, default=0.0,
                        help="Default base layer height is 0mm.")
    parser.add_argument("--infill_base", type=int, default=15,
                        help="Default % infill for the base shape model.")
    args = parser.parse_args()

    # 1) Parse Hueforge instructions from the TXT
    hf_data = parse_hueforge_txt(
        txt_path=args.hueforge_instructions,
        default_hueforge_layer_height=0.08,  # from your #3
        default_colors=None  # fallback to (1=Black,2=Silver,3=Gray,4=Jade White)
    )

    # Overwrite defaults from CLI if set
    base_lh = args.base_layer_height
    if abs(hf_data['base_layer_height'] - 0.2) > 1e-5:
        base_lh = hf_data['base_layer_height']  # user’s instructions override CLI

    hue_lh = hf_data['hueforge_layer_height']
    infill_base = args.infill_base if hf_data['infill_base'] == 15 else hf_data['infill_base']
    infill_hue = args.infill_hueforge if hf_data['infill_hueforge'] == 100 else hf_data['infill_hueforge']

    # If missing min_depth or max_depth, default to 0 and some large number
    min_depth = hf_data['min_depth'] if hf_data['min_depth'] is not None else 0.0
    max_depth = hf_data['max_depth'] if hf_data['max_depth'] is not None else 10.0

    # 2) base_model_height -> embed_offset
    # By your spec #8: embed_offset = base shape model height
    # If you have partial overlap, you might do (base_model_height - overlap).
    embed_offset = args.base_model_height

    # 3) Insert the config data
    update_3mf_metadata_ranges(
        three_mf_path=args.combined_3mf,
        base_layer_height=base_lh,
        hueforge_layer_height=hue_lh,
        min_depth=min_depth,
        max_depth=max_depth,
        infill_base=infill_base,
        infill_hueforge=infill_hue,
        base_model_height=embed_offset,  # #8
        hueforge_swaps=hf_data['swaps'],
        color_map=hf_data['color_map'],
        base_extruder=1,
        start_extruder_hueforge=2
    )

    print("All done. Check your final .3mf in Bambu Studio.")

if __name__ == "__main__":
    main()
