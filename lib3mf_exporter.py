"""
lib3mf Exporter Module for Bambu Studio Compatible 3MF Files

This module provides an alternative export path using the lib3mf C++ library
to generate 3MF files with Production Extension structure that Bambu Studio
can properly read, including component-based object structure.
"""

import os
import subprocess
import tempfile
import json
import zipfile
import shutil
from typing import Optional, Dict, Any
from pathlib import Path
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom


class Lib3mfExporter:
    """Export trimesh objects to Bambu-compatible 3MF using lib3mf C++ CLI."""

    def __init__(self, lib3mf_path: str = "/usr/local/bin/lib3mf-cli"):
        """
        Initialize the lib3mf exporter.

        Args:
            lib3mf_path: Path to the lib3mf CLI executable
        """
        self.lib3mf_path = lib3mf_path

    def check_lib3mf_available(self) -> bool:
        """Check if lib3mf CLI is available and working."""
        try:
            result = subprocess.run(
                [self.lib3mf_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def export_bambu_3mf(self,
                        mesh,  # trimesh.Trimesh object
                        output_path: str,
                        color_data: Optional[Dict] = None,
                        verbose: bool = True) -> bool:
        """
        Export a trimesh object to Bambu Studio compatible 3MF format.

        This method:
        1. Exports the mesh to a temporary STL file
        2. Uses lib3mf CLI to convert to 3MF with Production Extension
        3. Post-processes to add Bambu-specific structure
        4. Injects color metadata if provided

        Args:
            mesh: trimesh.Trimesh object to export
            output_path: Output path for the 3MF file
            color_data: Optional color layer data dictionary
            verbose: Print progress messages

        Returns:
            bool: True if successful, False otherwise
        """
        if verbose:
            print("🔧 Using lib3mf exporter for Bambu Studio compatibility")

        # Check if lib3mf is available
        if not self.check_lib3mf_available():
            print("❌ Error: lib3mf CLI not found. Falling back to standard export.")
            return False

        try:
            # Create temporary directory for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_stl = os.path.join(temp_dir, "mesh.stl")
                temp_3mf = os.path.join(temp_dir, "output.3mf")

                # Step 1: Export mesh to STL
                if verbose:
                    print("  📦 Exporting mesh to temporary STL...")
                mesh.export(temp_stl, file_type='stl_ascii')

                # Step 2: Convert STL to 3MF using lib3mf
                if verbose:
                    print("  🔄 Converting to 3MF with Production Extension...")

                if not self._convert_stl_to_3mf(temp_stl, temp_3mf, verbose):
                    return False

                # Step 3: Post-process for Bambu structure
                if verbose:
                    print("  🏗️ Adding Bambu Studio structure...")

                if not self._add_bambu_structure(temp_3mf, output_path, mesh, verbose):
                    return False

                # Step 4: Inject color metadata if provided
                if color_data:
                    if verbose:
                        print("  🎨 Injecting color layer metadata...")
                    self._inject_color_metadata_bambu(output_path, color_data, verbose)

                if verbose:
                    print("✅ Successfully exported Bambu-compatible 3MF")
                return True

        except Exception as e:
            print(f"❌ Error during Bambu 3MF export: {str(e)}")
            return False

    def _convert_stl_to_3mf(self, stl_path: str, output_path: str, verbose: bool) -> bool:
        """
        Use lib3mf CLI to convert STL to 3MF with Production Extension.

        Args:
            stl_path: Path to input STL file
            output_path: Path for output 3MF file
            verbose: Print command output

        Returns:
            bool: True if successful
        """
        cmd = [
            self.lib3mf_path,
            "convert",
            stl_path,
            output_path,
            "--production"  # Enable Production Extension
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                print(f"❌ lib3mf conversion failed: {result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            print("❌ lib3mf conversion timed out")
            return False
        except Exception as e:
            print(f"❌ lib3mf conversion error: {str(e)}")
            return False

    def _add_bambu_structure(self, input_3mf: str, output_3mf: str,
                             mesh, verbose: bool) -> bool:
        """
        Add Bambu Studio specific structure to the 3MF file.

        This restructures the 3MF to use component-based references
        as expected by Bambu Studio's Production Extension parser.

        Args:
            input_3mf: Path to lib3mf-generated 3MF
            output_3mf: Final output path
            mesh: Original trimesh object (for metadata)
            verbose: Print progress

        Returns:
            bool: True if successful
        """
        try:
            # Extract the 3MF file
            extract_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(input_3mf, 'r') as zf:
                zf.extractall(extract_dir)

            # Modify the 3D model XML structure
            model_path = os.path.join(extract_dir, '3D', '3dmodel.model')
            if os.path.exists(model_path):
                if not self._restructure_model_xml(model_path, mesh):
                    return False

            # Add Bambu-specific metadata files
            self._add_bambu_metadata_files(extract_dir)

            # Re-create the 3MF file
            with zipfile.ZipFile(output_3mf, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(extract_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_path = os.path.relpath(file_path, extract_dir)
                        zf.write(file_path, arc_path)

            # Cleanup
            shutil.rmtree(extract_dir)
            return True

        except Exception as e:
            print(f"❌ Error restructuring for Bambu: {str(e)}")
            return False

    def _restructure_model_xml(self, model_path: str, mesh) -> bool:
        """
        Restructure the 3D model XML to match Bambu's expected format.

        Args:
            model_path: Path to 3dmodel.model file
            mesh: Original trimesh object

        Returns:
            bool: True if successful
        """
        try:
            # Parse the existing model
            tree = ET.parse(model_path)
            root = tree.getroot()

            # Ensure proper namespaces
            ns = {
                '': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02',
                'p': 'http://schemas.microsoft.com/3dmanufacturing/production/2015/06'
            }

            # Register namespaces for output
            for prefix, uri in ns.items():
                if prefix:
                    ET.register_namespace(prefix, uri)
                else:
                    ET.register_namespace('', uri)

            # Find or create resources and build elements
            resources = root.find('.//resources', ns)
            if resources is None:
                resources = ET.SubElement(root, 'resources')

            build = root.find('.//build', ns)
            if build is None:
                build = ET.SubElement(root, 'build')

            # Ensure we have an object element (mesh container)
            objects = resources.findall('.//object', ns)
            if not objects:
                # Create a new object with mesh
                obj = ET.SubElement(resources, 'object', {
                    'id': '1',
                    'type': 'model'
                })
                # Note: Mesh data should already be in the file from lib3mf

            # Ensure build item references the object
            items = build.findall('.//item', ns)
            if not items:
                ET.SubElement(build, 'item', {
                    'objectid': '1'
                })

            # Add Production Extension component if needed
            components = resources.findall('.//components/component', ns)
            if not components:
                comps = ET.SubElement(resources, '{%s}components' % ns['p'])
                ET.SubElement(comps, '{%s}component' % ns['p'], {
                    'objectid': '1'
                })

            # Pretty print and save
            xml_str = ET.tostring(root, encoding='unicode')
            dom = minidom.parseString(xml_str)
            pretty_xml = dom.toprettyxml(indent="  ")

            # Remove extra blank lines
            lines = [line for line in pretty_xml.split('\n') if line.strip()]

            with open(model_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))

            return True

        except Exception as e:
            print(f"❌ Error restructuring model XML: {str(e)}")
            return False

    def _add_bambu_metadata_files(self, extract_dir: str):
        """
        Add Bambu Studio specific metadata files.

        Args:
            extract_dir: Directory containing extracted 3MF contents
        """
        metadata_dir = os.path.join(extract_dir, 'Metadata')
        os.makedirs(metadata_dir, exist_ok=True)

        # Add slice_info.config to identify as Bambu-compatible
        slice_info = {
            "slice_info": {
                "software": "SmithForge-WebUI",
                "compatible_printers": ["Bambu Lab X1", "Bambu Lab P1P", "Bambu Lab A1"]
            }
        }

        slice_info_path = os.path.join(metadata_dir, 'slice_info.config')
        with open(slice_info_path, 'w') as f:
            json.dump(slice_info, f, indent=2)

    def _inject_color_metadata_bambu(self, output_path: str,
                                     color_data: Dict, verbose: bool):
        """
        Inject color layer metadata into Bambu-compatible 3MF.

        Args:
            output_path: Path to 3MF file
            color_data: Color layer information
            verbose: Print progress
        """
        # This reuses the existing color injection logic
        # but ensures compatibility with Bambu structure
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract 3MF
                with zipfile.ZipFile(output_path, 'r') as zf:
                    zf.extractall(temp_dir)

                # Add color metadata files
                metadata_dir = os.path.join(temp_dir, 'Metadata')
                os.makedirs(metadata_dir, exist_ok=True)

                # Add layer_config_ranges.xml
                if 'layers' in color_data:
                    ranges_path = os.path.join(metadata_dir, 'layer_config_ranges.xml')
                    self._create_layer_ranges_xml(ranges_path, color_data['layers'])

                # Add project_settings.config
                if 'filament_colours' in color_data:
                    settings_path = os.path.join(metadata_dir, 'project_settings.config')
                    self._create_project_settings(settings_path, color_data['filament_colours'])

                # Re-zip
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for root, _, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arc_path = os.path.relpath(file_path, temp_dir)
                            zf.write(file_path, arc_path)

        except Exception as e:
            print(f"⚠️ Warning: Could not inject color metadata: {str(e)}")

    def _create_layer_ranges_xml(self, path: str, layers: list):
        """Create Bambu layer_config_ranges.xml file."""
        root = ET.Element('ranges')

        for i, layer in enumerate(layers):
            range_elem = ET.SubElement(root, 'range')
            range_elem.set('top_z', str(layer.get('top_z', 0)))
            range_elem.set('extruder', str(layer.get('extruder', i+1)))

            if 'color' in layer:
                range_elem.set('color', layer['color'])

        tree = ET.ElementTree(root)
        tree.write(path, encoding='utf-8', xml_declaration=True)

    def _create_project_settings(self, path: str, colors: list):
        """Create Bambu project_settings.config file."""
        settings = {
            "project_settings": {
                "filament_colour": colors
            }
        }

        with open(path, 'w') as f:
            json.dump(settings, f, indent=2)


def test_lib3mf_exporter():
    """Test function for the lib3mf exporter."""
    import trimesh

    # Create a simple test mesh
    mesh = trimesh.creation.box(extents=[10, 10, 10])

    # Test color data
    color_data = {
        'layers': [
            {'top_z': 0.0, 'extruder': '1', 'color': '#000000'},
            {'top_z': 1.0, 'extruder': '2', 'color': '#0047AB'},
            {'top_z': 2.0, 'extruder': '3', 'color': '#FFDA03'},
        ],
        'filament_colours': ['#000000', '#0047AB', '#FFDA03']
    }

    # Export
    exporter = Lib3mfExporter()
    success = exporter.export_bambu_3mf(
        mesh,
        'test_bambu.3mf',
        color_data=color_data
    )

    if success:
        print("✅ Test export successful!")
    else:
        print("❌ Test export failed!")

    return success


if __name__ == "__main__":
    test_lib3mf_exporter()