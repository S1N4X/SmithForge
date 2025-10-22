"""
SmithForge Text Layer Parser
Parses HueForge swap instruction text format to extract color layer information

Expected format:
    Filaments Used:
    PLA BambuLab Basic Black
    PLA BambuLab Basic Cobalt Blue
    ...

    Swap Instructions:
    Start with Black
    At layer #8 (0.72mm) swap to Cobalt Blue
    At layer #15 (1.28mm) swap to Sunflower Yellow
"""

import re
from typing import Dict, List, Optional, Tuple


# Color name to hex mapping (extended mapping for common filament colors)
COLOR_MAP = {
    # Basic colors
    'black': '#000000',
    'white': '#FFFFFF',
    'red': '#FF0000',
    'green': '#00FF00',
    'blue': '#0000FF',
    'yellow': '#FFFF00',
    'orange': '#FFA500',
    'purple': '#800080',
    'pink': '#FFC0CB',
    'brown': '#8B4513',
    'gray': '#808080',
    'grey': '#808080',

    # BambuLab PLA Basic colors (30 colors)
    'jade white': '#F0FFF0',
    'silver': '#C0C0C0',
    'light gray': '#D3D3D3',
    'dark gray': '#696969',
    'hot pink': '#FF69B4',
    'maroon red': '#800000',
    'beige': '#F5F5DC',
    'sunflower yellow': '#FFDA03',
    'gold': '#FFD700',
    'pumpkin orange': '#FF7518',
    'bambu green': '#00A86B',
    'mistletoe green': '#50C878',
    'bright green': '#66FF00',
    'blue grey': '#6699CC',
    'cobalt blue': '#0047AB',
    'turquoise': '#40E0D0',
    'indigo purple': '#4B0082',
    'bronze': '#CD7F32',
    'cocoa brown': '#D2691E',

    # BambuLab PLA Matte colors (25 colors)
    'matte ivory white': '#FFFFF0',
    'ivory white': '#FFFFF0',  # Also accept without "matte"
    'matte ash gray': '#B2BEB5',
    'ash gray': '#B2BEB5',
    'matte charcoal': '#36454F',
    'charcoal': '#36454F',
    'matte bone white': '#F9F6EE',
    'bone white': '#F9F6EE',
    'matte nardo gray': '#7A7D7A',
    'nardo gray': '#7A7D7A',
    'matte lemon yellow': '#FFF700',
    'lemon yellow': '#FFF700',
    'matte desert tan': '#C19A6B',
    'desert tan': '#C19A6B',
    'matte mandarin orange': '#FF8C00',
    'mandarin orange': '#FF8C00',
    'matte scarlet red': '#FF2400',
    'scarlet red': '#FF2400',
    'matte dark red': '#8B0000',
    'dark red': '#8B0000',
    'matte terracotta': '#CC4125',
    'terracotta': '#CC4125',
    'matte sakura pink': '#FFB7C5',
    'sakura pink': '#FFB7C5',
    'matte plum': '#DDA0DD',
    'plum': '#DDA0DD',
    'matte lilac purple': '#C8A2C8',
    'lilac purple': '#C8A2C8',
    'lilac': '#C8A2C8',
    'matte grass green': '#7CFC00',
    'grass green': '#7CFC00',
    'matte apple green': '#8DB600',
    'apple green': '#8DB600',
    'matte dark green': '#013220',
    'dark green': '#013220',
    'matte ice blue': '#99FFFF',
    'ice blue': '#99FFFF',
    'matte sky blue': '#87CEEB',
    'sky blue': '#87CEEB',
    'matte marine blue': '#0080FF',
    'marine blue': '#0080FF',
    'matte dark blue': '#00008B',
    'dark blue': '#00008B',
    'matte latte brown': '#B5651D',
    'latte brown': '#B5651D',
    'matte dark brown': '#654321',
    'dark brown': '#654321',
    'matte dark chocolate': '#3B2F2F',
    'dark chocolate': '#3B2F2F',
    'matte caramel': '#FFD59A',
    'caramel': '#FFD59A',

    # Common filament colors
    'transparent': '#FFFFFF80',
    'clear': '#FFFFFF80',
    'cyan': '#00FFFF',
    'magenta': '#FF00FF',
    'lime': '#00FF00',
    'navy': '#000080',
    'teal': '#008080',
    'maroon': '#800000',
    'olive': '#808000',

    # Legacy mappings (kept for compatibility)
    'basic black': '#000000',
    'basic white': '#FFFFFF',
}


def normalize_color_name(color_name: str) -> str:
    """
    Normalize color name by removing brand prefixes and converting to lowercase.

    Args:
        color_name: Original color name (e.g., "PLA BambuLab Basic Black")

    Returns:
        Normalized color name (e.g., "black")
    """
    # Remove common prefixes
    name = re.sub(r'^(PLA|ABS|PETG|TPU)\s+', '', color_name, flags=re.IGNORECASE)
    name = re.sub(r'(BambuLab|Bambu Lab|Prusament|Hatchbox|eSun)\s+', '', name, flags=re.IGNORECASE)
    name = re.sub(r'(Basic|Matte|Glossy|Silk|Metallic)\s+', '', name, flags=re.IGNORECASE)

    return name.strip().lower()


def color_name_to_hex(color_name: str) -> str:
    """
    Convert color name to hex code with smart fallback.

    Args:
        color_name: Color name (e.g., "Black", "Cobalt Blue")

    Returns:
        Hex color code (e.g., "#000000")
    """
    normalized = normalize_color_name(color_name)

    # Try exact match first
    if normalized in COLOR_MAP:
        return COLOR_MAP[normalized]

    # Try partial matches for compound names
    for color_key, hex_code in COLOR_MAP.items():
        if color_key in normalized or normalized in color_key:
            return hex_code

    # Smart fallback based on keywords in the color name
    fallback_colors = {
        'red': '#FF0000',
        'blue': '#0000FF',
        'green': '#00FF00',
        'yellow': '#FFFF00',
        'orange': '#FFA500',
        'purple': '#800080',
        'pink': '#FFC0CB',
        'brown': '#8B4513',
        'white': '#FFFFFF',
        'black': '#000000',
        'gray': '#808080',
        'grey': '#808080',
        'gold': '#FFD700',
        'silver': '#C0C0C0',
        'bronze': '#CD7F32',
        'cyan': '#00FFFF',
        'magenta': '#FF00FF',
        'teal': '#008080',
        'navy': '#000080',
        'maroon': '#800000',
        'olive': '#808000',
        'turquoise': '#40E0D0',
        'violet': '#8A2BE2',
        'lime': '#00FF00',
        'beige': '#F5F5DC',
        'tan': '#D2B48C',
    }

    lower = normalized.lower()
    for keyword, fallback_hex in fallback_colors.items():
        if keyword in lower:
            print(f"⚠️  Using generic {keyword} for unrecognized color '{color_name}'")
            return fallback_hex

    # Ultimate fallback to gray
    print(f"⚠️  Warning: Could not map color '{color_name}' to hex, using gray")
    return '#808080'


def parse_swap_instructions(text: str) -> Optional[Dict]:
    """
    Parse HueForge swap instruction text format.

    Args:
        text: Multi-line string containing filament list and swap instructions

    Returns:
        Dictionary compatible with extract_color_layers() format:
        {
            'layers': [{'top_z': float, 'extruder': str, 'color': str}, ...],
            'filament_colours': [hex_color_str, ...],
        }
        Or None if parsing fails
    """
    lines = text.strip().split('\n')

    # Track parsing state
    in_filaments_section = False
    in_swaps_section = False

    filaments: List[str] = []
    filament_colors: List[str] = []
    layers: List[Dict] = []

    # Track current extruder number
    current_extruder = 1

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Section headers
        if re.match(r'Filaments?\s+Used:', line, re.IGNORECASE):
            in_filaments_section = True
            in_swaps_section = False
            continue

        if re.match(r'Swap\s+Instructions?:', line, re.IGNORECASE):
            in_filaments_section = False
            in_swaps_section = True
            continue

        # Parse filament list
        if in_filaments_section:
            # Match lines like "PLA BambuLab Basic Black"
            # Should start with material type or brand name
            if re.match(r'^(PLA|ABS|PETG|TPU|Bambu|Prusa|Hatchbox)', line, re.IGNORECASE):
                filaments.append(line)
                hex_color = color_name_to_hex(line)
                filament_colors.append(hex_color)

        # Parse swap instructions
        if in_swaps_section:
            # Match "Start with <color>"
            start_match = re.match(r'Start\s+with\s+(.+)', line, re.IGNORECASE)
            if start_match:
                color_name = start_match.group(1).strip()
                # First layer starts at 0mm (will be adjusted later)
                layers.append({
                    'top_z': 0.0,
                    'extruder': str(current_extruder),
                    'color': color_name_to_hex(color_name)
                })
                current_extruder += 1
                continue

            # Match "At layer #N (X.XXmm) swap to <color>"
            swap_match = re.match(
                r'At\s+layer\s+#(\d+)\s+\(([0-9.]+)mm\)\s+swap\s+to\s+(.+)',
                line,
                re.IGNORECASE
            )
            if swap_match:
                layer_num = int(swap_match.group(1))
                z_height = float(swap_match.group(2))
                color_name = swap_match.group(3).strip()

                layers.append({
                    'top_z': z_height,
                    'extruder': str(current_extruder),
                    'color': color_name_to_hex(color_name)
                })
                current_extruder += 1
                continue

    # Validate we got something useful
    if not layers:
        print("❌ Error: No swap instructions found in text")
        return None

    if not filament_colors:
        print("⚠️  Warning: No filaments listed, using colors from swap instructions")
        # Generate filament colors from the layers
        filament_colors = [layer['color'] for layer in layers]

    print(f"✅ Parsed {len(layers)} layer swaps and {len(filament_colors)} filaments")

    return {
        'layers': layers,
        'filament_colours': filament_colors
    }


def validate_layer_heights(color_data: Dict, max_height: float) -> bool:
    """
    Validate that all layer heights are within the model bounds.

    Args:
        color_data: Dictionary from parse_swap_instructions()
        max_height: Maximum allowed Z height (final model height)

    Returns:
        True if all layers are valid, False otherwise (with warnings printed)
    """
    if not color_data or 'layers' not in color_data:
        return False

    all_valid = True

    for i, layer in enumerate(color_data['layers']):
        z = layer['top_z']

        if z < 0:
            print(f"⚠️  Warning: Layer {i+1} has negative Z-height: {z:.3f}mm")
            all_valid = False

        if z > max_height:
            print(f"⚠️  Warning: Layer {i+1} Z-height ({z:.3f}mm) exceeds model height ({max_height:.3f}mm)")
            all_valid = False

    return all_valid


if __name__ == "__main__":
    # Test the parser with example input
    example_text = """
    Filaments Used:
    PLA BambuLab Basic Black
    PLA BambuLab Basic Cobalt Blue
    PLA BambuLab Basic Sunflower Yellow
    PLA BambuLab Matte Ivory White

    Swap Instructions:
    Start with Black
    At layer #8 (0.72mm) swap to Cobalt Blue
    At layer #15 (1.28mm) swap to Sunflower Yellow
    At layer #22 (2.00mm) swap to Ivory White
    """

    result = parse_swap_instructions(example_text)

    if result:
        print("\n=== Parsed Result ===")
        print(f"Filament colors: {result['filament_colours']}")
        print(f"\nLayers:")
        for layer in result['layers']:
            print(f"  Z={layer['top_z']:.3f}mm, Color={layer['color']}, Extruder={layer['extruder']}")
