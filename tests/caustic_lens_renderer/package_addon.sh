#!/bin/bash
# Package the Caustic Lens Renderer add-on for distribution
#
# Usage:
#   ./package_addon.sh           # Package for legacy Blender (3.x/4.0/4.1)
#   ./package_addon.sh --ext     # Package for Blender 4.2+ Extensions
#
# Output:
#   caustic_lens_renderer-X.Y.Z.zip

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Extract version from __init__.py
VERSION=$(grep -oP '"version":\s*\(\K[0-9]+,\s*[0-9]+,\s*[0-9]+' __init__.py | tr -d ' ' | tr ',' '.')

if [ -z "$VERSION" ]; then
    VERSION="1.0.0"
fi

PACKAGE_NAME="caustic_lens_renderer-${VERSION}.zip"

echo "Packaging Caustic Lens Renderer v${VERSION}..."

if [ "$1" == "--ext" ]; then
    # Blender 4.2+ Extensions format
    echo "Building for Blender 4.2+ Extensions..."
    
    # Check if blender is available
    if command -v blender &> /dev/null; then
        blender --command extension build --source-dir "$SCRIPT_DIR" --output-dir "$SCRIPT_DIR"
        echo "Extension package created with Blender's extension builder"
    else
        echo "Blender not found in PATH. Creating manual package..."
        # Manual package (folder structure)
        rm -f "$PACKAGE_NAME"
        zip -r "$PACKAGE_NAME" __init__.py blender_manifest.toml
        echo "Created: $SCRIPT_DIR/$PACKAGE_NAME"
    fi
else
    # Legacy format (single __init__.py in root of zip)
    echo "Building for legacy Blender (3.x/4.0/4.1)..."
    rm -f "$PACKAGE_NAME"
    
    # Create a temp directory with folder structure
    TEMP_DIR=$(mktemp -d)
    mkdir -p "$TEMP_DIR/caustic_lens_renderer"
    cp __init__.py "$TEMP_DIR/caustic_lens_renderer/"
    
    # Create zip from temp directory
    cd "$TEMP_DIR"
    zip -r "$SCRIPT_DIR/$PACKAGE_NAME" caustic_lens_renderer
    cd "$SCRIPT_DIR"
    rm -rf "$TEMP_DIR"
    
    echo "Created: $SCRIPT_DIR/$PACKAGE_NAME"
fi

echo ""
echo "Installation instructions:"
echo "  1. Open Blender"
echo "  2. Go to Edit > Preferences > Add-ons"
echo "  3. Click 'Install...' button"
echo "  4. Select: $PACKAGE_NAME"
echo "  5. Enable 'Caustic Lens Renderer' in the add-on list"
echo "  6. Access via View3D sidebar (N) > Caustic tab"
