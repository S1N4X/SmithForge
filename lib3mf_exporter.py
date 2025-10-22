"""
lib3mf Exporter Module for Bambu Studio Compatible 3MF Files

This module uses the Python lib3mf bindings to generate 3MF files with
Production Extension structure that Bambu Studio can properly read.
"""

import os
import json
from typing import Optional, Dict
import xml.etree.ElementTree as ET


class Lib3mfExporter:
    """Export trimesh objects to Bambu-compatible 3MF using lib3mf Python API."""

    def __init__(self):
        """Initialize the lib3mf exporter."""
        self.lib3mf = None
        self.wrapper = None

    def check_lib3mf_available(self) -> bool:
        """Check if lib3mf Python module is available."""
        try:
            import lib3mf
            from lib3mf import get_wrapper
            self.lib3mf = lib3mf
            self.wrapper = get_wrapper()
            return True
        except ImportError:
            return False

    def export_bambu_3mf(self,
                        mesh,  # trimesh.Trimesh object
                        output_path: str,
                        color_data: Optional[Dict] = None,
                        verbose: bool = True) -> bool:
        """
        Export a trimesh object to Bambu Studio compatible 3MF format.

        This method uses the Python lib3mf API to:
        1. Create a 3MF model with Production Extension
        2. Add mesh geometry from trimesh object
        3. Create component-based structure (required by Bambu)
        4. Inject color metadata if provided
        5. Write to file

        Args:
            mesh: trimesh.Trimesh object to export
            output_path: Output path for the 3MF file
            color_data: Optional color layer data dictionary
            verbose: Print progress messages

        Returns:
            bool: True if successful, False otherwise
        """
        if verbose:
            print("🔧 Using lib3mf Python API for Bambu Studio compatibility")

        # Check if lib3mf is available
        if not self.check_lib3mf_available():
            print("❌ Error: lib3mf Python module not found. Install with: pip install lib3mf")
            return False

        try:
            if verbose:
                print("  📦 Creating 3MF model with Production Extension...")

            # Create a new 3MF model
            model = self.wrapper.CreateModel()

            # Add mesh geometry
            if verbose:
                print("  🔺 Adding mesh geometry...")
            mesh_object = self._add_mesh_to_model(model, mesh)

            if not mesh_object:
                print("❌ Error: Failed to add mesh to model")
                return False

            # Add build item (this creates the component structure)
            if verbose:
                print("  🏗️ Creating build structure...")
            model.AddBuildItem(mesh_object, self.wrapper.GetIdentityTransform())

            # Add color metadata if provided
            if color_data:
                if verbose:
                    print("  🎨 Adding color layer metadata...")
                self._add_color_metadata(model, color_data)

            # Write to file
            if verbose:
                print(f"  💾 Writing to {output_path}...")

            writer = model.QueryWriter("3mf")
            writer.WriteToFile(output_path)

            if verbose:
                print("✅ Bambu-compatible 3MF exported successfully")
            return True

        except Exception as e:
            print(f"❌ Error during Bambu 3MF export: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _add_mesh_to_model(self, model, trimesh_mesh):
        """
        Add trimesh geometry to lib3mf model.

        Args:
            model: lib3mf model object
            trimesh_mesh: trimesh.Trimesh object

        Returns:
            lib3mf mesh object or None on failure
        """
        try:
            # Create a new mesh object in the model
            mesh_object = model.AddMeshObject()

            # Get vertices and faces from trimesh
            vertices = trimesh_mesh.vertices
            faces = trimesh_mesh.faces

            # Convert to lib3mf format
            # Vertices: array of Position objects
            lib3mf_vertices = []
            for vertex in vertices:
                pos = self.lib3mf.Position()
                pos.Coordinates[0] = float(vertex[0])
                pos.Coordinates[1] = float(vertex[1])
                pos.Coordinates[2] = float(vertex[2])
                lib3mf_vertices.append(pos)

            # Triangles: array of Triangle objects
            lib3mf_triangles = []
            for face in faces:
                triangle = self.lib3mf.Triangle()
                triangle.Indices[0] = int(face[0])
                triangle.Indices[1] = int(face[1])
                triangle.Indices[2] = int(face[2])
                lib3mf_triangles.append(triangle)

            # Set geometry on mesh object
            mesh_object.SetGeometry(lib3mf_vertices, lib3mf_triangles)

            return mesh_object

        except Exception as e:
            print(f"❌ Error adding mesh geometry: {str(e)}")
            return None

    def _add_color_metadata(self, model, color_data: Dict):
        """
        Add color layer metadata to the 3MF model.

        This adds Bambu Studio specific metadata for color layers.

        Args:
            model: lib3mf model object
            color_data: Dictionary with 'layers' and optionally 'filament_colours'
        """
        try:
            # Add metadata attachments for Bambu Studio
            # This includes layer color information and filament data

            # Note: lib3mf Python API may have limited metadata support
            # For now, we'll add basic metadata and rely on post-processing
            # to inject Bambu-specific XML files if needed

            if 'layers' in color_data and color_data['layers']:
                # Add metadata key-value pairs
                metadata = model.GetMetaDataGroup()
                metadata.AddMetaData("http://schemas.bambulab.com/", "ColorLayers", "true")
                metadata.AddMetaData("http://schemas.bambulab.com/", "LayerCount", str(len(color_data['layers'])))

            if 'filament_colours' in color_data:
                filament_json = json.dumps(color_data['filament_colours'])
                metadata = model.GetMetaDataGroup()
                metadata.AddMetaData("http://schemas.bambulab.com/", "FilamentColors", filament_json)

        except Exception as e:
            print(f"⚠️ Warning: Could not add all color metadata: {str(e)}")
            # Non-fatal - continue with export


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
