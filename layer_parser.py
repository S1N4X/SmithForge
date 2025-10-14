"""
SmithForge Layer Parser
Extracts color layer information from 3MF files for visualization

Note: When visualizing layers after embedding, remember to account for
DEFAULT_EMBEDDING_OVERLAP_MM from smithforge.py (currently 0.1mm).
"""

import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
import os

# Import the embedding constant for reference in layer adjustments
try:
    from smithforge import DEFAULT_EMBEDDING_OVERLAP_MM
except ImportError:
    # Fallback if module not found (shouldn't happen in normal usage)
    DEFAULT_EMBEDDING_OVERLAP_MM = 0.1


class LayerInfo:
    """Represents a single layer with color and height information"""
    def __init__(self, z_height: float, color: Optional[str] = None,
                 layer_number: Optional[int] = None):
        self.z_height = z_height
        self.color = color
        self.layer_number = layer_number

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'z_height': self.z_height,
            'color': self.color,
            'layer_number': self.layer_number
        }


def parse_3mf_layers(file_path: str) -> Dict:
    """
    Parse layer information from a 3MF file.

    Args:
        file_path: Path to the 3MF file

    Returns:
        Dictionary containing layer information:
        {
            'layers': [LayerInfo, ...],
            'total_height': float,
            'layer_count': int,
            'has_colors': bool,
            'format': str  # 'bambu', 'prusa', 'hueforge', or 'unknown'
        }
    """
    result = {
        'layers': [],
        'total_height': 0.0,
        'layer_count': 0,
        'has_colors': False,
        'format': 'unknown'
    }

    if not os.path.exists(file_path):
        return result

    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # Try different metadata formats

            # Try Bambu Lab format (layer_config_ranges.xml)
            if 'Metadata/layer_config_ranges.xml' in zip_ref.namelist():
                result = _parse_bambu_layers(zip_ref)
                result['format'] = 'bambu'

            # Try PrusaSlicer/SuperSlicer format (config metadata in model file)
            elif '3D/3dmodel.model' in zip_ref.namelist():
                # Try to extract metadata from model file
                model_data = _parse_model_file(zip_ref)
                if model_data:
                    result = model_data
                    result['format'] = 'prusa'

            # Try generic color info from 3dmodel.model
            if result['layer_count'] == 0:
                generic_data = _parse_generic_colors(zip_ref)
                if generic_data and generic_data['layer_count'] > 0:
                    result = generic_data
                    result['format'] = 'generic'

    except Exception as e:
        print(f"Error parsing 3MF layers: {e}")
        result['error'] = str(e)

    return result


def _parse_bambu_layers(zip_ref: zipfile.ZipFile) -> Dict:
    """
    Parse Bambu Lab layer_config_ranges.xml format.
    This file contains layer ranges with color change information.
    """
    layers = []

    try:
        xml_content = zip_ref.read('Metadata/layer_config_ranges.xml')
        root = ET.fromstring(xml_content)

        # Parse layer ranges
        for range_elem in root.findall('.//range'):
            min_z = float(range_elem.get('minZ', 0))
            max_z = float(range_elem.get('maxZ', 0))

            # Try to extract color information
            color = None
            color_elem = range_elem.find('.//filament_colour')
            if color_elem is not None and color_elem.text:
                color = color_elem.text.strip()

            # Create layer for this range (use max_z as the layer height)
            if max_z > 0:
                layer = LayerInfo(z_height=max_z, color=color)
                layers.append(layer)

        # Sort by z height
        layers.sort(key=lambda l: l.z_height)

        # Assign layer numbers
        for i, layer in enumerate(layers):
            layer.layer_number = i + 1

        total_height = layers[-1].z_height if layers else 0.0
        has_colors = any(l.color for l in layers)

        return {
            'layers': layers,
            'total_height': total_height,
            'layer_count': len(layers),
            'has_colors': has_colors
        }

    except Exception as e:
        print(f"Error parsing Bambu layers: {e}")
        return {'layers': [], 'total_height': 0.0, 'layer_count': 0, 'has_colors': False}


def _parse_model_file(zip_ref: zipfile.ZipFile) -> Optional[Dict]:
    """
    Parse the 3dmodel.model file for metadata.
    This may contain custom metadata from various slicers.
    """
    try:
        model_content = zip_ref.read('3D/3dmodel.model')
        root = ET.fromstring(model_content)

        # Define namespace
        ns = {'model': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}

        # Try to find metadata section
        metadata = root.find('.//model:metadata', ns)
        if metadata is not None:
            # Look for layer-related metadata
            # This varies by slicer, so we check multiple possible formats
            pass

        # For now, return None as we need specific examples to implement
        return None

    except Exception as e:
        print(f"Error parsing model file: {e}")
        return None


def _parse_generic_colors(zip_ref: zipfile.ZipFile) -> Dict:
    """
    Parse generic color information from the model file.
    Extracts colors assigned to objects/components.
    """
    layers = []

    try:
        model_content = zip_ref.read('3D/3dmodel.model')
        root = ET.fromstring(model_content)

        # Define namespaces
        ns = {
            'model': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02',
            'materials': 'http://schemas.microsoft.com/3dmanufacturing/material/2015/02'
        }

        # Find color resources
        resources = root.find('.//model:resources', ns)
        if resources is not None:
            # Look for color groups or base materials
            for basematerials in resources.findall('.//materials:basematerials', ns):
                for base in basematerials.findall('.//materials:base', ns):
                    color = base.get('displaycolor')
                    name = base.get('name', '')

                    if color:
                        # We don't have Z height info from just colors
                        # So we create placeholder layers
                        layer = LayerInfo(z_height=0.0, color=color)
                        layers.append(layer)

        has_colors = len(layers) > 0

        return {
            'layers': layers,
            'total_height': 0.0,
            'layer_count': len(layers),
            'has_colors': has_colors
        }

    except Exception as e:
        print(f"Error parsing generic colors: {e}")
        return {'layers': [], 'total_height': 0.0, 'layer_count': 0, 'has_colors': False}


def adjust_layers_for_zshift(layers: List[LayerInfo], z_shift: float) -> List[LayerInfo]:
    """
    Adjust layer Z heights for a given Z shift (for visualization after embedding).

    Args:
        layers: List of LayerInfo objects
        z_shift: Z offset to apply

    Returns:
        New list of LayerInfo objects with adjusted heights
    """
    adjusted = []
    for layer in layers:
        new_layer = LayerInfo(
            z_height=layer.z_height + z_shift,
            color=layer.color,
            layer_number=layer.layer_number
        )
        adjusted.append(new_layer)
    return adjusted


def serialize_layers(layer_data: Dict) -> Dict:
    """
    Serialize layer data for JSON response.

    Args:
        layer_data: Dictionary from parse_3mf_layers

    Returns:
        JSON-serializable dictionary
    """
    result = {
        'layers': [layer.to_dict() for layer in layer_data.get('layers', [])],
        'total_height': layer_data.get('total_height', 0.0),
        'layer_count': layer_data.get('layer_count', 0),
        'has_colors': layer_data.get('has_colors', False),
        'format': layer_data.get('format', 'unknown')
    }

    if 'error' in layer_data:
        result['error'] = layer_data['error']

    return result
