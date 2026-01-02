#!/bin/bash
# Convert WSL paths to Windows paths and run native Blender
# Note: Output is saved to a temp Windows path then copied back to WSL

BLENDER_WIN="/mnt/c/Program Files/Blender Foundation/Blender 5.0/blender.exe"

# Convert paths to Windows format
SCRIPT=$(wslpath -w "$1")
BLEND=$(wslpath -w "$2")
OBJ=$(wslpath -w "$3")

# For output, use a temp file in Windows temp directory to avoid UNC path issues
# Extract just the filename from the output path
OUTPUT_FILENAME=$(basename "$4")
WIN_TEMP_OUTPUT="C:\\Temp\\${OUTPUT_FILENAME}"
WSL_TEMP_OUTPUT="/mnt/c/Temp/${OUTPUT_FILENAME}"

# Create Windows temp directory if it doesn't exist
mkdir -p /mnt/c/Temp

# Run Blender with the Windows temp output path
"$BLENDER_WIN" --background --python "$SCRIPT" -- "$BLEND" "$OBJ" "$WIN_TEMP_OUTPUT"

# Copy the rendered file back to the intended WSL location
if [ -f "$WSL_TEMP_OUTPUT" ]; then
    cp "$WSL_TEMP_OUTPUT" "$4"
    rm "$WSL_TEMP_OUTPUT"
    echo "Output copied to: $4"
else
    echo "ERROR: Render output not found at $WSL_TEMP_OUTPUT"
    exit 1
fi