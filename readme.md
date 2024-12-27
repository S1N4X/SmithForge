# ForgeSmith
Python script to combine two 3MF models by overlaying one (Hueforge) onto another (base) with automatic scaling, positioning, and proper model intersection.

This script was initially written to easily embed existing HueForge models on top of ChromaLink hexagon bases. (TODO: Insert link)

## Features
- Automatic scaling to match base dimensions while maintaining aspect ratio
- Base model rotation support
- Custom X/Y/Z positioning
- Automatic centering
- Proper model intersection and union
- Smart Z-axis embedding with configurable overlap
- Maintains model integrity through convex hull clipping

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
- Default overlap is 0.2mm for proper model union
- Scale maintains aspect ratio in X/Y dimensions
- Z-axis height is never modified (as per HueForge conventions)
- Models must have proper manifold geometry

## Future Features
- Add height modifiers and filament colours from HueForge description files in the 3MF output

## Requirements
- Python 3.x
- trimesh
- numpy
- shapely
- lib3mf

## Installation
```bash
pip install trimesh numpy shapely lib3mf
```

## Usage
```bash
python put_3mf_on_3mf.py -f hueforge.3mf -b base.3mf [options]
```

### Parameters
Required:
- -f, --hueforge: Path to Hueforge 3MF file to be placed on top
- -b, --base: Path to base 3MF file

Optional:
- -o, --output: Output file path (default: combined.3mf)
- --rotatebase: Rotate base model (degrees, 0-360)
- -s, --scale: Force specific scale value
- --scaledown: Allow downscaling below 1.0
- --xshift: X-axis shift in mm
- --yshift: Y-axis shift in mm
- --zshift: Z-axis shift in mm

### Examples
Basic combination:
```bash
python put_3mf_on_3mf.py -f design.3mf -b base.3mf
```

Rotate base and adjust position of HueForge (in mm):
```bash
python put_3mf_on_3mf.py -f design.3mf -b base.3mf --rotatebase 90 --xshift 5 --zshift 0.5
```

Force a user defined scale for the HueForge model (example: 1.5x size)
```bash
python put_3mf_on_3mf.py -f design.3mf -b base.3mf -s 1.5
```