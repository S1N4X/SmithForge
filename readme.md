<p align="center">
  <img src="img\forgesmith_logo_white_small_v2.gif" alt="ForgeSmith Logo">
</p>

ForgeSmith is a tool for seamlessly combining two 3MF models by overlaying and embedding a Hueforge model onto a base shape with automatic scaling, positioning, and precise intersection alignment. The resulting 3MF file is ready to be opened in your favourite slicer for 3D printing. 


<p align="center">
  <img src="img/process.png" alt="Process">
</p>

This small but efficient script can be used to automate the process of shaping HueForges models to your needs. HueForge Z axis integrity is kept through the process (which means the script does not modify the original Z height config).

This project was originally designed to simplify embedding Hueforge models onto ChromaLink (previously named MagHex) hexagonal bases. Please see the [Slug3D.com](https://www.slug3d.com) ChromaLink's framework for 3D printed wall art designing.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Features
- Automatic scaling of X & Y axis to match the base dimensions while maintaining aspect ratio.
- The Z-axis scaling remains unchanged throughout the process to avoid disrupting the Hueforge's visual integrity
- Also maintains model integrity through convex hull clipping
- Automatic centering
- Custom X/Y/Z positioning
- Proper model intersection and union
- Smart Z-axis embedding with configurable overlap
- Base model rotation support

## Technical Process
1. Loads both 3MF models
2. Optionally rotates base model
3. Calculates and applies scaling to match base dimensions
4. Centers Hueforge on base model
5. Embeds Hueforge with slight Z overlap
6. Applies user-specified position adjustments
7. Creates convex hull of base for proper intersection
8. Performs boolean operations for clean combination
9. Exports final unified model

## Notes
- Default overlap is 0.1mm for proper model union (can be changed with --zshift)
- Scale maintains aspect ratio in X/Y dimensions
- Z-axis height is never modified (as per HueForge guidelines)
- Models must have proper manifold geometry

## Future Features
- Add height modifiers and filament colours from HueForge description files in the 3MF output

## Requirements
- Python 3.x
- trimesh
- shapely
- A base model (.3mf)
- A model to be embedded like a HueForge (.3mf)

## Installation
```bash
pip install trimesh shapely
```

## Usage
```bash
python forgesmith.py -f hueforge.3mf -b base.3mf [options]
```

### Parameters
#### File paths
```bash
-f, --hueforge: Path of the 3MF file that will serve as the overlay to be embedded to the base (required)
-b, --base: Path of the base shape 3MF file (required)
-o, --output: Output file path (default: combined.3mf)
```

#### Overlay geometry
```bash
--xshift: X-axis shift in mm
--yshift: Y-axis shift in mm
--zshift: Z-axis shift in mm
-s, --scale: Force specific scale value
--scaledown: Allow downscaling below 1.0
```

#### Base geometry
```bash
--rotatebase: Rotate base model (degrees, 0-360)
```

### Examples
Basic combination:
```bash
python forgesmith.py -f hueforge.3mf -b base.3mf
```

Rotate base and adjust position of HueForge (in mm):
```bash
python forgesmith.py -f hueforge.3mf -b base.3mf --rotatebase 90 --xshift 5 --zshift 0.5
```

Force a user defined scale for the HueForge model (example: 1.5x size)
```bash
python forgesmith.py -f hueforge.3mf -b base.3mf -s 1.5
```