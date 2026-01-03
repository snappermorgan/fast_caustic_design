# AI Agent Instructions for Fast Caustic Design

This file provides context and guidelines for AI coding agents working on this codebase.

## Project Overview
A **freeform optics designer** that generates 3D lens surfaces creating specific light patterns (caustics). Based on the paper ["Instant Transport Maps on 2D Grids"](https://hal.inria.fr/hal-01884157) by Nader & Guennebaud.

**Core Pipeline** (in `apps/caustic_design.cpp`):
1. **Optimal Transport (OT)**: Computes `T(x)` mapping light from lens to target plane via `runOptimalTransport()`
2. **Normal Integration**: Iteratively refines surface normals (10 outer iterations) via `fresnelMapping()` + `normal_int.perform_normal_integration()`

## Architecture

| Component | Location | License | Purpose |
|-----------|----------|---------|---------|
| Main Pipeline | `apps/caustic_design.cpp` | GPL3 | CLI entry, orchestrates OT → geometry |
| OT Solver | `otlib/otsolver_2dgrid.h` | GPL3 | `GridBasedTransportSolver` - Monge-Ampère on quad grid |
| Normal Integration | `apps/normal_integration/` | MPL2 | Ceres-based surface reconstruction |
| Mesh | `apps/normal_integration/mesh.h` | MPL2 | Triangle/quad mesh generation |
| Utilities | `apps/common/` | MPL2 | Image I/O (CImg), procedural patterns, CLI parsing |

## Build & Run

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release  # Add Cholmod for 10x+ speedup
cmake --build .
./caustic_design -in_trg pattern.png -res 128 -focal_l 1.5
```

**Key Dependencies**: Eigen3 (bundled), Ceres Solver (required), CImg (bundled), libpng/libjpeg, SuiteSparse/Cholmod (optional but highly recommended).

**Windows (vcpkg)**: Use `-DCMAKE_TOOLCHAIN_FILE=...\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows-static`

## Testing & Debugging

- **Procedural patterns** (no image needed): `-in_trg :8:256:` (pattern 8 = circles at 256px)
  - Pattern IDs in `apps/common/analytical_functions.h`: `CIRCLES=8`, `SINGLE_BLACK=10`, `SINGLE_WHITE=11`, etc.
- **Verbose output**: `-v 10` shows solver residuals/iterations
- **Convergence tuning**: `-th 1e-7` (threshold), `-itr 1000` (max iterations)

## Critical Conventions

### Image Processing
- **Orientation**: All images pass through `rotate90ClockwiseAndFlipX()` before processing
- **Normalization**: Scaled to `[0, 1]` via `scaleAndTranslate()`

### Coordinate System
- **Mesh domain**: `[0, 1]²` with margin `mesh_width/resolution`
- **Focal plane**: Target points at `z = -focal_length`
- **Z-snapping**: Surface translated so `max_z = 0` each outer iteration

### Cost Function Weights (`apps/normal_integration/costFunctor.h`)
```cpp
#define EINT_WEIGHT 0.01   // Normal integration term
#define EDIR_WEIGHT 5.0    // Direction matching term  
#define EREG_WEIGHT 1.0    // Regularization term
```

## Extending the Codebase

| Task | Files to Modify |
|------|-----------------|
| New procedural pattern | Add to `FuncName::Enum` in `analytical_functions.h`, implement in `.cpp` |
| New cost function | Add functor class in `costFunctor.h`, wire in `normal_integration.cpp` |
| Custom OT solver | Implement interface matching `GridBasedTransportSolver` in `otlib/` |

## Known Limitations
- **OT solver assumes square domain** - only square lenses are supported
- **Refractive index hardcoded**: `double r = 1.55` in `caustic_design.cpp`
- **Composition heuristic**: `T(u→v) = T(u→1) ∘ T(1→v)` is approximate, not true L2 optimal transport

## Automated Testing Workflow

When iterating on codebase changes, follow this workflow:

### Quick Test Command
```bash
./tests/test_workflow.sh 200   # Fast iteration (res=200)
./tests/test_workflow.sh 500   # Final quality (res=500)
```

### Step-by-Step Workflow
1. **Make changes** to the codebase (stay within project folder only)
2. **Build**: `cmake --build build`
3. **Generate mesh**:
   ```bash
   ./build/caustic_design -in_trg tests/jkHeart.png -focal_l 3 -mesh_width 1 -res 200 -output tests/jkHeart.obj
   ```
4. **Render test pattern**:
   ```bash
   ./tests/winblender.sh tests/renderBlender.py tests/CausticTemplate.blend tests/jkHeart.obj tests/render.png
   ```
5. **Compare** `tests/jkHeart.png` (input) vs `tests/render.png` (output)
6. **Present findings** and wait for approval before next iteration

### VS Code Tasks
- `test-full-workflow` - Complete build→mesh→render cycle (res=200)
- `test-full-workflow-hires` - High quality cycle (res=500)
- `generate-mesh-only` - Just generate the mesh
- `render-only` - Just render existing mesh

### Success Criteria
The rendered caustic pattern in `tests/render.png` should match the input image `tests/jkHeart.png`. Look for:
- Shape accuracy (heart outline preserved)
- Light distribution (bright/dark regions match)
- Edge sharpness (no excessive blurring)
- No artifacts (swirls, discontinuities)

## Scientific Context
**Brenier's Theorem**: Optimal transport map = gradient of convex potential (curl-free), enabling accurate normal integration. The composition heuristic can introduce small curl, causing lens artifacts.
