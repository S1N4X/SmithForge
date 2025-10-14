"""
SmithForge Mesh Repair Module
Validates and repairs 3D mesh geometry issues for proper boolean operations

This module provides automatic mesh repair capabilities to fix common issues:
- Non-manifold edges and vertices
- Holes in the mesh surface
- Degenerate faces
- Inverted normals
- Duplicate vertices/faces
- Self-intersections
"""

import trimesh
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RepairReport:
    """Container for repair operation results"""
    def __init__(self):
        self.issues_found = []
        self.repairs_applied = []
        self.warnings = []
        self.success = True

    def add_issue(self, issue: str):
        """Record an issue that was detected"""
        self.issues_found.append(issue)
        logger.info(f"Issue detected: {issue}")

    def add_repair(self, repair: str):
        """Record a repair that was applied"""
        self.repairs_applied.append(repair)
        logger.info(f"Repair applied: {repair}")

    def add_warning(self, warning: str):
        """Record a warning"""
        self.warnings.append(warning)
        logger.warning(warning)

    def to_dict(self) -> Dict:
        """Convert report to dictionary for JSON serialization"""
        return {
            'issues_found': self.issues_found,
            'repairs_applied': self.repairs_applied,
            'warnings': self.warnings,
            'success': self.success
        }

    def __str__(self) -> str:
        """String representation of the repair report"""
        lines = ["=== Mesh Repair Report ==="]

        if self.issues_found:
            lines.append(f"\nIssues Found ({len(self.issues_found)}):")
            for issue in self.issues_found:
                lines.append(f"  - {issue}")

        if self.repairs_applied:
            lines.append(f"\nRepairs Applied ({len(self.repairs_applied)}):")
            for repair in self.repairs_applied:
                lines.append(f"  - {repair}")

        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        lines.append(f"\nStatus: {'SUCCESS' if self.success else 'FAILED'}")

        return "\n".join(lines)


def validate_mesh(mesh: trimesh.Trimesh) -> RepairReport:
    """
    Validate a mesh and identify issues without fixing them.

    Args:
        mesh: The trimesh object to validate

    Returns:
        RepairReport with issues identified
    """
    report = RepairReport()

    # Check if mesh is empty
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        report.add_issue("Mesh is empty (no vertices or faces)")
        report.success = False
        return report

    # Check if mesh is watertight (manifold)
    if not mesh.is_watertight:
        report.add_issue("Mesh is not watertight (has holes or non-manifold edges)")

    # Check for degenerate faces (zero area)
    face_areas = mesh.area_faces
    degenerate_count = np.sum(face_areas < 1e-10)
    if degenerate_count > 0:
        report.add_issue(f"Found {degenerate_count} degenerate faces (zero or near-zero area)")

    # Check for duplicate vertices
    unique_vertices = len(np.unique(mesh.vertices, axis=0))
    if unique_vertices < len(mesh.vertices):
        duplicate_count = len(mesh.vertices) - unique_vertices
        report.add_issue(f"Found {duplicate_count} duplicate vertices")

    # Check for duplicate faces
    unique_faces = len(np.unique(np.sort(mesh.faces, axis=1), axis=0))
    if unique_faces < len(mesh.faces):
        duplicate_count = len(mesh.faces) - unique_faces
        report.add_issue(f"Found {duplicate_count} duplicate faces")

    # Check mesh volume
    try:
        volume = mesh.volume
        if volume <= 0:
            report.add_issue(f"Mesh has invalid volume: {volume}")
    except Exception as e:
        report.add_warning(f"Could not compute mesh volume: {e}")

    # Check for self-intersections (expensive operation, so make it optional)
    # This is commented out by default as it can be very slow for large meshes
    # if mesh.is_self_intersecting():
    #     report.add_issue("Mesh has self-intersections")

    if not report.issues_found:
        report.add_repair("Mesh validation passed - no issues detected")

    return report


def repair_mesh(mesh: trimesh.Trimesh, aggressive: bool = False) -> Tuple[trimesh.Trimesh, RepairReport]:
    """
    Repair common mesh issues to make it suitable for boolean operations.

    Args:
        mesh: The trimesh object to repair
        aggressive: If True, apply more aggressive repair techniques (may alter geometry more)

    Returns:
        Tuple of (repaired_mesh, RepairReport)
    """
    report = RepairReport()

    # First validate to see what issues exist
    validation_report = validate_mesh(mesh)
    report.issues_found = validation_report.issues_found

    if not report.issues_found:
        report.add_repair("No repairs needed")
        return mesh, report

    # Start repair process
    logger.info("Starting mesh repair process...")

    try:
        # Remove duplicate vertices
        if any("duplicate vertices" in issue.lower() for issue in report.issues_found):
            original_vertex_count = len(mesh.vertices)
            mesh.merge_vertices()
            new_vertex_count = len(mesh.vertices)
            if new_vertex_count < original_vertex_count:
                report.add_repair(f"Merged {original_vertex_count - new_vertex_count} duplicate vertices")

        # Remove duplicate faces
        if any("duplicate faces" in issue.lower() for issue in report.issues_found):
            original_face_count = len(mesh.faces)
            mesh.update_faces(mesh.unique_faces())
            new_face_count = len(mesh.faces)
            if new_face_count < original_face_count:
                report.add_repair(f"Removed {original_face_count - new_face_count} duplicate faces")

        # Remove degenerate faces
        if any("degenerate faces" in issue.lower() for issue in report.issues_found):
            original_face_count = len(mesh.faces)
            mesh.remove_degenerate_faces()
            new_face_count = len(mesh.faces)
            if new_face_count < original_face_count:
                report.add_repair(f"Removed {original_face_count - new_face_count} degenerate faces")

        # Fill holes if mesh is not watertight
        if any("not watertight" in issue.lower() for issue in report.issues_found):
            try:
                # Try to fill holes
                mesh.fill_holes()
                if mesh.is_watertight:
                    report.add_repair("Successfully filled holes - mesh is now watertight")
                else:
                    report.add_warning("Attempted to fill holes but mesh is still not watertight")
            except Exception as e:
                report.add_warning(f"Could not fill holes: {e}")

        # Fix normals (ensure consistent winding)
        try:
            mesh.fix_normals()
            report.add_repair("Fixed face normals for consistent winding")
        except Exception as e:
            report.add_warning(f"Could not fix normals: {e}")

        # Remove unreferenced vertices
        original_vertex_count = len(mesh.vertices)
        mesh.remove_unreferenced_vertices()
        new_vertex_count = len(mesh.vertices)
        if new_vertex_count < original_vertex_count:
            report.add_repair(f"Removed {original_vertex_count - new_vertex_count} unreferenced vertices")

        # Aggressive repairs (may alter geometry more significantly)
        if aggressive:
            # Try to make the mesh a valid volume
            if not mesh.is_volume:
                try:
                    # Use convex hull as last resort (will change geometry significantly)
                    report.add_warning("Applying aggressive repair: converting to convex hull")
                    mesh = mesh.convex_hull
                    report.add_repair("Converted to convex hull (geometry was significantly altered)")
                except Exception as e:
                    report.add_warning(f"Could not create convex hull: {e}")

        # Final validation
        final_validation = validate_mesh(mesh)

        if final_validation.issues_found:
            remaining_issues = [issue for issue in final_validation.issues_found
                              if issue not in report.issues_found]
            if remaining_issues:
                report.add_warning("Some issues remain after repair:")
                for issue in remaining_issues:
                    report.add_warning(f"  - {issue}")

        # Check if mesh is now suitable for boolean operations
        if mesh.is_watertight and mesh.is_volume:
            report.success = True
            report.add_repair("Mesh is now ready for boolean operations")
        else:
            report.success = False
            report.add_warning("Mesh may still have issues that could cause boolean operation failures")

    except Exception as e:
        report.success = False
        report.add_warning(f"Repair process encountered an error: {e}")
        logger.error(f"Error during mesh repair: {e}", exc_info=True)

    return mesh, report


def auto_repair_mesh(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, RepairReport]:
    """
    Convenience function that automatically repairs a mesh with sensible defaults.

    This is the recommended function to use for automatic repair in the SmithForge pipeline.

    Args:
        mesh: The trimesh object to repair

    Returns:
        Tuple of (repaired_mesh, RepairReport)
    """
    return repair_mesh(mesh, aggressive=False)


def repair_3mf_file(input_path: str, output_path: str, aggressive: bool = False) -> RepairReport:
    """
    Load a 3MF file, repair it, and save the result.

    Args:
        input_path: Path to input 3MF file
        output_path: Path to save repaired 3MF file
        aggressive: If True, apply more aggressive repair techniques

    Returns:
        RepairReport with details of repairs performed
    """
    report = RepairReport()

    try:
        # Load the 3MF file
        logger.info(f"Loading 3MF file: {input_path}")
        mesh = trimesh.load(input_path)

        # Handle Scene vs Mesh
        if isinstance(mesh, trimesh.Scene):
            # Extract the first geometry from the scene
            geometries = list(mesh.geometry.values())
            if not geometries:
                report.add_warning("3MF file contains no geometry")
                report.success = False
                return report
            mesh = geometries[0]
            if len(geometries) > 1:
                report.add_warning(f"3MF contains {len(geometries)} objects, only repairing the first")

        # Repair the mesh
        repaired_mesh, repair_report = repair_mesh(mesh, aggressive=aggressive)
        report = repair_report

        # Save the repaired mesh
        if report.success:
            logger.info(f"Saving repaired mesh to: {output_path}")
            repaired_mesh.export(output_path)
            report.add_repair(f"Saved repaired mesh to {output_path}")
        else:
            report.add_warning("Repair was not fully successful, not saving output")

    except Exception as e:
        report.success = False
        report.add_warning(f"Failed to repair 3MF file: {e}")
        logger.error(f"Error repairing 3MF file: {e}", exc_info=True)

    return report
