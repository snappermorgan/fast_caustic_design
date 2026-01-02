#!/bin/bash
# Automated test workflow for caustic design
# Usage: ./tests/test_workflow.sh [resolution]
#   resolution: mesh resolution (default: 200 for faster iteration, use 500 for final)

set -e

PROJECT_DIR="/home/cmorgan/fast_caustic_design"
TEST_IMAGE="tests/jkHeart.png"
OUTPUT_OBJ="tests/jkHeart.obj"
OUTPUT_RENDER="tests/render.png"
RESOLUTION="${1:-200}"  # Default to 200 for faster iteration

cd "$PROJECT_DIR"

echo "=============================================="
echo "CAUSTIC DESIGN TEST WORKFLOW"
echo "=============================================="
echo "Input image:  $TEST_IMAGE"
echo "Resolution:   $RESOLUTION"
echo "=============================================="

echo ""
echo "=== Step 1: Building project ==="
cmake --build "$PROJECT_DIR/build"
if [ $? -ne 0 ]; then
    echo "ERROR: Build failed!"
    exit 1
fi
echo "Build successful."

echo ""
echo "=== Step 2: Generating caustic mesh ==="
"$PROJECT_DIR/build/caustic_design" \
    -in_trg "$PROJECT_DIR/$TEST_IMAGE" \
    -focal_l 3 \
    -mesh_width 1 \
    -res "$RESOLUTION" \
    -output "$PROJECT_DIR/$OUTPUT_OBJ"

if [ ! -f "$PROJECT_DIR/$OUTPUT_OBJ" ]; then
    echo "ERROR: Mesh generation failed - output file not created!"
    exit 1
fi
echo "Mesh generated: $OUTPUT_OBJ"

echo ""
echo "=== Step 3: Rendering in Blender ==="
"$PROJECT_DIR/tests/winblender.sh" \
    "$PROJECT_DIR/tests/renderBlender.py" \
    "$PROJECT_DIR/tests/CausticTemplate.blend" \
    "$PROJECT_DIR/$OUTPUT_OBJ" \
    "$PROJECT_DIR/$OUTPUT_RENDER"

if [ ! -f "$PROJECT_DIR/$OUTPUT_RENDER" ]; then
    echo "ERROR: Blender render failed - output file not created!"
    exit 1
fi
echo "Render complete: $OUTPUT_RENDER"

echo ""
echo "=============================================="
echo "TEST WORKFLOW COMPLETE"
echo "=============================================="
echo "Input image:   $PROJECT_DIR/$TEST_IMAGE"
echo "Output render: $PROJECT_DIR/$OUTPUT_RENDER"
echo ""
echo "Compare these images to evaluate the caustic quality."
echo "The projected light pattern should match the input image."
echo "=============================================="
