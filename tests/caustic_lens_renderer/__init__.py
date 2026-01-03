"""
Caustic Lens Renderer - Blender Add-on

Generate caustic lens meshes from target images and render caustic patterns.

Supports:
- Windows executable (.exe) on Windows
- Linux executable on Linux
- Linux executable via WSL from Windows
"""

bl_info = {
    "name": "Caustic Lens Renderer",
    "author": "Fast Caustic Design Contributors",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Caustic",
    "description": "Generate caustic lens meshes from images and render caustic patterns",
    "warning": "",
    "doc_url": "https://github.com/cmorgan/fast_caustic_design",
    "tracker_url": "https://github.com/cmorgan/fast_caustic_design/issues",
    "category": "3D View",
}

import bpy
import os
import sys
import subprocess
import threading
import re
import json
import platform
from bpy.props import StringProperty, FloatProperty, IntProperty, BoolProperty, EnumProperty, PointerProperty


# Global state for progress tracking
class GenerationState:
    is_running = False
    progress = 0.0
    status_text = "Idle"
    process = None
    output_lines = []

gen_state = GenerationState()


def get_platform_info():
    """Detect current platform and available execution methods."""
    info = {
        'system': platform.system(),  # 'Windows', 'Linux', 'Darwin'
        'is_windows': platform.system() == 'Windows',
        'is_linux': platform.system() == 'Linux',
        'is_wsl_available': False,
        'wsl_distro': None
    }
    
    # Check for WSL on Windows
    if info['is_windows']:
        try:
            result = subprocess.run(
                ['wsl', '--list', '--quiet'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                distros = [d.strip() for d in result.stdout.strip().split('\n') if d.strip()]
                if distros:
                    info['is_wsl_available'] = True
                    info['wsl_distro'] = distros[0]  # Use first available distro
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass
    
    return info


def detect_executable_type(executable_path):
    """Detect if executable is Windows (.exe) or Linux binary."""
    if not executable_path:
        return None
    
    # Normalize path
    exec_lower = executable_path.lower()
    
    # Check extension
    if exec_lower.endswith('.exe'):
        return 'windows'
    
    # Check for WSL path format (\\wsl$ or \\wsl.localhost)
    if exec_lower.startswith('\\\\wsl'):
        return 'linux_wsl'
    
    # Check for Linux-style path
    if executable_path.startswith('/'):
        return 'linux'
    
    # Try to read file header to detect ELF (Linux) vs PE (Windows)
    try:
        with open(executable_path, 'rb') as f:
            header = f.read(4)
            if header[:4] == b'\x7fELF':
                return 'linux'
            elif header[:2] == b'MZ':
                return 'windows'
    except Exception:
        pass
    
    # Default based on current platform
    platform_info = get_platform_info()
    return 'windows' if platform_info['is_windows'] else 'linux'


def convert_path_for_wsl(windows_path):
    """Convert Windows path to WSL path format."""
    if not windows_path:
        return windows_path
    
    # Already a WSL/Linux path
    if windows_path.startswith('/'):
        return windows_path
    
    # Handle \\wsl.localhost\ or \\wsl$\ paths
    if windows_path.lower().startswith('\\\\wsl'):
        # \\wsl.localhost\Ubuntu\home\user\file -> /home/user/file
        # \\wsl$\Ubuntu\home\user\file -> /home/user/file
        parts = windows_path.split('\\')
        # Skip \\wsl.localhost\distro or \\wsl$\distro (first 4 parts)
        if len(parts) > 4:
            return '/' + '/'.join(parts[4:])
        return windows_path
    
    # Handle standard Windows paths (C:\path\to\file)
    if len(windows_path) >= 2 and windows_path[1] == ':':
        drive = windows_path[0].lower()
        rest = windows_path[2:].replace('\\', '/')
        return f'/mnt/{drive}{rest}'
    
    return windows_path


def convert_path_for_windows(wsl_path):
    """Convert WSL/Linux path to Windows accessible path."""
    if not wsl_path:
        return wsl_path
    
    # Already a Windows path
    if '\\' in wsl_path or (len(wsl_path) >= 2 and wsl_path[1] == ':'):
        return wsl_path
    
    # Handle /mnt/c/ style paths
    if wsl_path.startswith('/mnt/') and len(wsl_path) > 5:
        drive = wsl_path[5].upper()
        rest = wsl_path[6:].replace('/', '\\')
        return f'{drive}:{rest}'
    
    # For other Linux paths, assume they're in default WSL distro
    # This requires knowing the distro name
    platform_info = get_platform_info()
    if platform_info['is_wsl_available'] and platform_info['wsl_distro']:
        distro = platform_info['wsl_distro']
        wsl_path_windows = wsl_path.replace("/", "\\")
        return f'\\\\wsl.localhost\\{distro}{wsl_path_windows}'
    
    return wsl_path


def build_execution_command(executable, args, platform_info, exec_type):
    """Build the appropriate command based on platform and executable type."""
    
    if platform_info['is_linux']:
        # Running on Linux - execute directly
        return [executable] + args
    
    elif platform_info['is_windows']:
        if exec_type == 'windows':
            # Windows exe on Windows - execute directly
            return [executable] + args
        
        elif exec_type in ('linux', 'linux_wsl'):
            # Linux binary on Windows - use WSL
            if not platform_info['is_wsl_available']:
                raise RuntimeError("Linux executable requires WSL, but WSL is not available")
            
            # Convert executable path to WSL format
            wsl_exec = convert_path_for_wsl(executable)
            
            # Convert all path arguments to WSL format
            wsl_args = []
            for arg in args:
                # Check if argument looks like a path
                if os.path.sep in arg or arg.startswith('/') or (len(arg) > 1 and arg[1] == ':'):
                    wsl_args.append(convert_path_for_wsl(arg))
                else:
                    wsl_args.append(arg)
            
            return ['wsl', wsl_exec] + wsl_args
    
    # Fallback - try direct execution
    return [executable] + args


class CausticProperties(bpy.types.PropertyGroup):
    """Properties for caustic rendering setup."""
    
    # === Generation Properties ===
    caustic_executable: StringProperty(
        name="Executable",
        description="Path to caustic_design executable (.exe for Windows, or Linux binary)",
        default="",
        subtype='FILE_PATH'
    )  # type: ignore
    
    input_image: StringProperty(
        name="Input Image",
        description="Path to input target image (PNG)",
        default="",
        subtype='FILE_PATH'
    )  # type: ignore
    
    output_folder: StringProperty(
        name="Output Folder",
        description="Folder to save generated OBJ file",
        default="",
        subtype='DIR_PATH'
    )  # type: ignore
    
    output_filename: StringProperty(
        name="Output Filename",
        description="Name for the output OBJ file (without extension)",
        default="caustic_lens"
    )  # type: ignore
    
    gen_mesh_width: FloatProperty(
        name="Mesh Width",
        description="Width of the lens in output units (mm or inches)",
        default=76.2,
        min=0.1,
        max=1000.0
    )  # type: ignore
    
    gen_focal_length: FloatProperty(
        name="Focal Length",
        description="Distance from lens to projection plane",
        default=228.6,
        min=0.1,
        max=10000.0
    )  # type: ignore
    
    gen_thickness: FloatProperty(
        name="Thickness",
        description="Thickness of the lens",
        default=3.175,
        min=0.01,
        max=100.0
    )  # type: ignore
    
    gen_resolution: IntProperty(
        name="Resolution",
        description="Mesh resolution (higher = more detail, slower)",
        default=200,
        min=50,
        max=1000
    )  # type: ignore
    
    gen_padding: FloatProperty(
        name="Padding",
        description="Padding around image as fraction (0.21 for circular cutout safety)",
        default=0.0,
        min=0.0,
        max=0.5
    )  # type: ignore
    
    gen_verbose: IntProperty(
        name="Verbosity",
        description="Verbosity level (0-10)",
        default=5,
        min=0,
        max=10
    )  # type: ignore


    
    # === Render Properties ===
    obj_path: StringProperty(
        name="OBJ File",
        description="Path to the caustic lens OBJ file",
        default="",
        subtype='FILE_PATH'
    )  # type: ignore
    
    focal_length: FloatProperty(
        name="Focal Length",
        description="Distance from lens to projection plane (mm)",
        default=228.6,
        min=1.0,
        max=10000.0
    )  # type: ignore
    
    lens_width: FloatProperty(
        name="Lens Width",
        description="Width of the lens (mm)",
        default=76.2,
        min=1.0,
        max=1000.0
    )  # type: ignore
    
    light_distance_multiplier: FloatProperty(
        name="Light Distance (x lens width)",
        description="How far the light is from the lens, as multiple of lens width",
        default=40.0,
        min=10.0,
        max=100.0
    )  # type: ignore
    
    light_energy: FloatProperty(
        name="Light Energy",
        description="Light power/energy",
        default=20000.0,
        min=0.1,
        max=100000.0
    )  # type: ignore
    
    glass_ior: FloatProperty(
        name="Glass IOR",
        description="Index of refraction for glass material",
        default=1.5,
        min=1.0,
        max=3.0
    )  # type: ignore
    
    render_samples: IntProperty(
        name="Render Samples",
        description="Number of render samples",
        default=256,
        min=32,
        max=4096
    )  # type: ignore


def parse_progress(line):
    """Parse caustic_design output to determine progress."""
    global gen_state
    
    # Store output line
    gen_state.output_lines.append(line)
    if len(gen_state.output_lines) > 100:
        gen_state.output_lines.pop(0)
    
    # Try parsing as JSON first
    if line.strip().startswith("{"):
        try:
            data = json.loads(line)
            if "progress" in data:
                gen_state.progress = float(data["progress"])
            if "message" in data:
                gen_state.status_text = data["message"]
            if "status" in data:
                status = data["status"]
                # Map internal status to user readable if needed
                if status == "processing":
                    pass 
            return
        except json.JSONDecodeError:
            pass # Fallback to text parsing

    # Parse different stages (Fallback for text output)
    if "Loading target image" in line or "Reading input" in line:
        gen_state.progress = 0.05
        gen_state.status_text = "Loading image..."
    elif "Image dimensions" in line:
        gen_state.progress = 0.10
        gen_state.status_text = "Image loaded"
    elif "Applying padding" in line:
        gen_state.progress = 0.12
        gen_state.status_text = "Applying padding..."
    elif "Running OT solver" in line or "Optimal Transport" in line:
        gen_state.progress = 0.15
        gen_state.status_text = "Running OT solver..."
    elif "OT iteration" in line or "otsolver" in line.lower():
        # Try to parse iteration number
        match = re.search(r'(\d+)\s*/\s*(\d+)', line)
        if match:
            current, total = int(match.group(1)), int(match.group(2))
            gen_state.progress = 0.15 + 0.30 * (current / max(total, 1))
        gen_state.status_text = "OT solving..."
    elif "Transport map computed" in line or "OT complete" in line:
        gen_state.progress = 0.45
        gen_state.status_text = "OT complete"
    elif "Normal integration" in line or "outer iteration" in line.lower():
        # Parse outer iteration
        match = re.search(r'outer\s*iteration\s*(\d+)', line, re.IGNORECASE)
        if match:
            iteration = int(match.group(1))
            gen_state.progress = 0.50 + 0.04 * iteration  # 10 iterations = 40%
            gen_state.status_text = f"Normal integration {iteration}/10..."
        else:
            gen_state.progress = 0.50
            gen_state.status_text = "Normal integration..."
    elif "Ceres" in line or "iterations" in line.lower():
        gen_state.status_text = "Optimizing surface..."
    elif "Saving" in line or "Writing" in line or "Exporting" in line:
        gen_state.progress = 0.95
        gen_state.status_text = "Saving mesh..."
    elif "Done" in line or "Complete" in line or "finished" in line.lower():
        gen_state.progress = 1.0
        gen_state.status_text = "Complete!"


def run_caustic_design(executable, args, platform_info, exec_type, on_complete):
    """Run caustic_design in a background thread."""
    global gen_state
    
    try:
        gen_state.is_running = True
        gen_state.progress = 0.0
        gen_state.status_text = "Starting..."
        gen_state.output_lines = []
        
        # Build the appropriate command
        cmd = build_execution_command(executable, args, platform_info, exec_type)
        print(f"Platform: {platform_info['system']}, Executable type: {exec_type}")
        print(f"Running: {' '.join(cmd)}")
        
        # Set up environment
        env = os.environ.copy()
        
        # On Windows, we may need to handle WSL differently
        if platform_info['is_windows'] and exec_type in ('linux', 'linux_wsl'):
            # WSL execution
            gen_state.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
        else:
            # Direct execution
            gen_state.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )
        
        for line in gen_state.process.stdout:
            line = line.strip()
            if line:
                print(f"[caustic_design] {line}")
                parse_progress(line)
        
        gen_state.process.wait()
        
        if gen_state.process.returncode == 0:
            gen_state.progress = 1.0
            gen_state.status_text = "Complete!"
        else:
            gen_state.status_text = f"Error (code {gen_state.process.returncode})"
            print(f"===== CAUSTIC_DESIGN FAILED (exit code {gen_state.process.returncode}) =====")
            print(f"Last output lines:")
            for line in gen_state.output_lines[-20:]:
                print(f"  {line}")
            print("=" * 50)
            
    except Exception as e:
        gen_state.status_text = f"Error: {str(e)}"
        print(f"Error running caustic_design: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gen_state.is_running = False
        gen_state.process = None
        if on_complete:
            on_complete()


class CAUSTIC_OT_generate_mesh(bpy.types.Operator):
    """Generate caustic lens mesh using caustic_design executable."""
    bl_idname = "caustic.generate_mesh"
    bl_label = "Generate Mesh"
    bl_options = {'REGISTER'}
    
    _timer = None
    
    def modal(self, context, event):
        global gen_state
        
        if event.type == 'TIMER':
            # Force UI redraw to show progress
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
            
            if not gen_state.is_running:
                self.cancel(context)
                
                if gen_state.progress >= 1.0:
                    # Auto-set the OBJ path for rendering
                    props = context.scene.caustic_props
                    output_path = os.path.join(
                        bpy.path.abspath(props.output_folder),
                        props.output_filename + ".obj"
                    )
                    props.obj_path = output_path
                    props.focal_length = props.gen_focal_length
                    props.lens_width = props.gen_mesh_width
                    
                    self.report({'INFO'}, f"Mesh generated: {output_path}")
                else:
                    self.report({'ERROR'}, f"Generation failed: {gen_state.status_text}")
                
                return {'FINISHED'}
        
        return {'PASS_THROUGH'}
    
    def execute(self, context):
        global gen_state
        
        if gen_state.is_running:
            self.report({'WARNING'}, "Generation already in progress")
            return {'CANCELLED'}
        
        props = context.scene.caustic_props
        platform_info = get_platform_info()
        
        # Validate executable
        executable = bpy.path.abspath(props.caustic_executable)
        if not executable:
            self.report({'ERROR'}, "No executable specified")
            return {'CANCELLED'}
        
        # Detect executable type
        exec_type = detect_executable_type(executable)
        print(f"Detected executable type: {exec_type}")
        
        # Validate executable exists
        # For WSL paths on Windows, we need special handling
        if platform_info['is_windows'] and exec_type in ('linux', 'linux_wsl'):
            # Convert to Windows-accessible path for existence check
            win_path = convert_path_for_windows(executable)
            if not os.path.exists(win_path) and not os.path.exists(executable):
                self.report({'ERROR'}, f"Executable not found: {executable}")
                return {'CANCELLED'}
        elif not os.path.exists(executable):
            self.report({'ERROR'}, f"Executable not found: {executable}")
            return {'CANCELLED'}
        
        # Check if WSL is needed but not available
        if platform_info['is_windows'] and exec_type in ('linux', 'linux_wsl'):
            if not platform_info['is_wsl_available']:
                self.report({'ERROR'}, "Linux executable requires WSL, but WSL is not installed")
                return {'CANCELLED'}
        
        # Validate input image
        input_image = bpy.path.abspath(props.input_image)
        if not input_image or not os.path.exists(input_image):
            self.report({'ERROR'}, f"Input image not found: {input_image}")
            return {'CANCELLED'}
        
        # Validate output folder
        output_folder = bpy.path.abspath(props.output_folder)
        if not output_folder or not os.path.isdir(output_folder):
            self.report({'ERROR'}, f"Output folder not found: {output_folder}")
            return {'CANCELLED'}
        
        output_path = os.path.join(output_folder, props.output_filename + ".obj")
        
        # Build command arguments
        args = [
            "-in_trg", input_image,
            "-output", output_path,
            "-mesh_width", str(props.gen_mesh_width),
            "-focal_l", str(props.gen_focal_length),
            "-thickness", str(props.gen_thickness),
            "-res", str(props.gen_resolution),
            "-v", str(props.gen_verbose),
        ]
            
        # Add JSON progress flag for parsing
        args.append("--json-progress")

        # Start background thread
        thread = threading.Thread(
            target=run_caustic_design,
            args=(executable, args, platform_info, exec_type, None)
        )
        thread.start()
        
        # Start modal timer
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        
        return {'RUNNING_MODAL'}
    
    def cancel(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None


class CAUSTIC_OT_cancel_generation(bpy.types.Operator):
    """Cancel the running generation process."""
    bl_idname = "caustic.cancel_generation"
    bl_label = "Cancel Generation"
    
    def execute(self, context):
        global gen_state
        
        if gen_state.process:
            gen_state.process.terminate()
            gen_state.status_text = "Cancelled"
            gen_state.is_running = False
            self.report({'INFO'}, "Generation cancelled")
        
        return {'FINISHED'}


class CAUSTIC_OT_detect_platform(bpy.types.Operator):
    """Detect platform and executable information."""
    bl_idname = "caustic.detect_platform"
    bl_label = "Detect Platform"
    
    def execute(self, context):
        props = context.scene.caustic_props
        platform_info = get_platform_info()
        
        msg_parts = [f"Platform: {platform_info['system']}"]
        
        if platform_info['is_windows']:
            if platform_info['is_wsl_available']:
                msg_parts.append(f"WSL available ({platform_info['wsl_distro']})")
            else:
                msg_parts.append("WSL not available")
        
        if props.caustic_executable:
            exec_type = detect_executable_type(bpy.path.abspath(props.caustic_executable))
            msg_parts.append(f"Executable type: {exec_type}")
        
        self.report({'INFO'}, " | ".join(msg_parts))
        return {'FINISHED'}


def import_obj_file(filepath):
    """Import OBJ file with compatibility for Blender 3.x and 4.x."""
    if bpy.app.version >= (4, 0, 0):
        # Blender 4.0+ uses wm.obj_import
        bpy.ops.wm.obj_import(filepath=filepath)
    else:
        # Blender 3.x uses import_scene.obj
        bpy.ops.import_scene.obj(filepath=filepath)


class CAUSTIC_OT_import_lens(bpy.types.Operator):
    """Import caustic lens OBJ and set up material."""
    bl_idname = "caustic.import_lens"
    bl_label = "Import Lens"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.caustic_props
        
        if not props.obj_path:
            self.report({'ERROR'}, "No OBJ file specified")
            return {'CANCELLED'}
        
        # Expand path
        obj_path = bpy.path.abspath(props.obj_path)
        
        if not os.path.exists(obj_path):
            self.report({'ERROR'}, f"File not found: {obj_path}")
            return {'CANCELLED'}
        
        # Remove existing lens if present
        if "CausticLens" in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects["CausticLens"], do_unlink=True)
        
        # Import OBJ (version-compatible)
        import_obj_file(obj_path)
        lens_obj = context.selected_objects[0]
        lens_obj.name = "CausticLens"
        
        # Apply smooth shading
        context.view_layer.objects.active = lens_obj
        bpy.ops.object.shade_smooth()
        
        # Position lens
        lens_obj.location = (props.lens_width / 2, -props.focal_length, props.lens_width / 2)
        
        # Enable caustics
        if bpy.app.version >= (4, 0, 0):
            lens_obj.cycles.is_caustics_caster = True
        else:
            if hasattr(lens_obj.cycles, 'cast_shadow_caustics'):
                lens_obj.cycles.cast_shadow_caustics = True
        
        # Create glass material
        self.setup_glass_material(lens_obj, props.glass_ior)
        
        self.report({'INFO'}, f"Imported lens: {lens_obj.name}")
        return {'FINISHED'}
    
    def setup_glass_material(self, obj, ior):
        """Create and assign glass material."""
        mat_name = "CausticGlass"
        
        # Remove existing material
        if mat_name in bpy.data.materials:
            bpy.data.materials.remove(bpy.data.materials[mat_name])
        
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        
        # Create Principled BSDF
        shader = nodes.new(type='ShaderNodeBsdfPrincipled')
        shader.location = (0, 0)
        shader.inputs['Roughness'].default_value = 0.0
        
        # Handle transmission input name difference between Blender versions
        if bpy.app.version >= (4, 0, 0):
            shader.inputs['Transmission Weight'].default_value = 1.0
        else:
            shader.inputs['Transmission'].default_value = 1.0
        
        shader.inputs['IOR'].default_value = ior
        
        # Create Output node
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (300, 0)
        
        # Link
        mat.node_tree.links.new(shader.outputs['BSDF'], output.inputs['Surface'])
        
        # Assign to object
        obj.data.materials.clear()
        obj.data.materials.append(mat)


class CAUSTIC_OT_setup_light(bpy.types.Operator):
    """Set up point light for caustic rendering."""
    bl_idname = "caustic.setup_light"
    bl_label = "Setup Light"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.caustic_props
        
        # Remove existing caustic lights
        for obj in list(bpy.data.objects):
            if obj.name.startswith("CausticLight"):
                bpy.data.objects.remove(obj, do_unlink=True)
        
        light_distance = props.lens_width * props.light_distance_multiplier
                
        light_data = bpy.data.lights.new(name="CausticLight", type='POINT')
        light_data.energy = props.light_energy
        light_data.shadow_soft_size = 0.0  # Set to zero for sharpest caustics
        
        # Enable Shadow Caustics on the light (try different API versions)
        if hasattr(light_data.cycles, 'is_caustics_light'):
            light_data.cycles.is_caustics_light = True
        elif hasattr(light_data.cycles, 'cast_shadow_caustics'):
            light_data.cycles.cast_shadow_caustics = True
        elif hasattr(light_data, 'use_shadow_caustics'):
            light_data.use_shadow_caustics = True           
        
        light_obj = bpy.data.objects.new("CausticLight", light_data)
        context.collection.objects.link(light_obj)
        
        # Position far behind lens
        light_obj.location = (props.lens_width / 2, -light_distance, props.lens_width / 2)
        
        self.report({'INFO'}, f"Created point light at distance {light_distance:.1f}mm")
        return {'FINISHED'}


class CAUSTIC_OT_setup_projection_plane(bpy.types.Operator):
    """Create projection plane at focal distance."""
    bl_idname = "caustic.setup_projection_plane"
    bl_label = "Setup Projection Plane"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.caustic_props
        
        # Remove existing projection plane
        if "ProjectionPlane" in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects["ProjectionPlane"], do_unlink=True)
        
        # Create plane
        bpy.ops.mesh.primitive_plane_add(size=props.lens_width * 2)
        plane = context.active_object
        plane.name = "ProjectionPlane"
        
        # Position at focal length
        from math import radians
        plane.location = (props.lens_width / 2, props.focal_length, props.lens_width / 2)
        plane.rotation_euler = (radians(90), 0, 0)  # Face the lens
        
        # Enable caustics receiver for the plane
        if bpy.app.version >= (4, 0, 0):
            plane.cycles.is_caustics_receiver = True
        
        # Create matte white material
        mat = bpy.data.materials.new(name="ProjectionMatte")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        shader = nodes.get("Principled BSDF")
        if shader:
            shader.inputs['Base Color'].default_value = (1, 1, 1, 1)
            shader.inputs['Roughness'].default_value = 1.0
        
        plane.data.materials.clear()
        plane.data.materials.append(mat)
        
        self.report({'INFO'}, f"Created projection plane at Y={props.focal_length:.1f}mm")
        return {'FINISHED'}


class CAUSTIC_OT_setup_render(bpy.types.Operator):
    """Configure render settings for caustics."""
    bl_idname = "caustic.setup_render"
    bl_label = "Setup Render Settings"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.caustic_props
        scene = context.scene
        
        # Use Cycles
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU'
        
        # GPU setup - try CUDA first, then other options
        prefs = context.preferences.addons['cycles'].preferences
        
        gpu_found = False
        for compute_type in ['CUDA', 'OPTIX', 'HIP', 'ONEAPI', 'METAL']:
            try:
                prefs.compute_device_type = compute_type
                prefs.get_devices()
                for device in prefs.devices:
                    if device.type == compute_type:
                        device.use = True
                        gpu_found = True
                if gpu_found:
                    break
            except Exception:
                continue
        
        if not gpu_found:
            scene.cycles.device = 'CPU'
            self.report({'WARNING'}, "No GPU found, using CPU")
        
        # Render settings
        scene.cycles.samples = props.render_samples
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.01
        
        # Denoiser
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = 'OPENIMAGEDENOISE'
        
        # Light bounces optimized for caustics
        scene.cycles.max_bounces = 8
        scene.cycles.diffuse_bounces = 2
        scene.cycles.glossy_bounces = 2
        scene.cycles.transmission_bounces = 8
        scene.cycles.transparent_max_bounces = 8
        
        self.report({'INFO'}, "Render settings configured for caustics")
        return {'FINISHED'}


class CAUSTIC_OT_full_setup(bpy.types.Operator):
    """Run complete caustic setup: import lens, setup light, plane, and render settings."""
    bl_idname = "caustic.full_setup"
    bl_label = "Full Setup"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        bpy.ops.caustic.import_lens()
        bpy.ops.caustic.setup_light()
        bpy.ops.caustic.setup_projection_plane()
        bpy.ops.caustic.setup_render()
        
        self.report({'INFO'}, "Full caustic setup complete!")
        return {'FINISHED'}


class CAUSTIC_PT_generate_panel(bpy.types.Panel):
    """Panel for mesh generation."""
    bl_label = "Generate Caustic Mesh"
    bl_idname = "CAUSTIC_PT_generate_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Caustic'
    bl_order = 0
    
    def draw(self, context):
        global gen_state
        layout = self.layout
        props = context.scene.caustic_props
        platform_info = get_platform_info()
        
        # Platform info
        box = layout.box()
        row = box.row()
        row.label(text=f"Platform: {platform_info['system']}", icon='SYSTEM')
        if platform_info['is_windows'] and platform_info['is_wsl_available']:
            row.label(text=f"WSL: {platform_info['wsl_distro']}", icon='CHECKMARK')
        
        # Executable path
        box = layout.box()
        box.label(text="Executable", icon='CONSOLE')
        box.prop(props, "caustic_executable", text="")
        
        # Show detected type
        if props.caustic_executable:
            exec_type = detect_executable_type(bpy.path.abspath(props.caustic_executable))
            exec_label = {
                'windows': 'Windows (.exe)',
                'linux': 'Linux binary',
                'linux_wsl': 'Linux (WSL path)'
            }.get(exec_type, 'Unknown')
            box.label(text=f"Type: {exec_label}", icon='INFO')
        
        # Input/Output
        box = layout.box()
        box.label(text="Input / Output", icon='FILE_IMAGE')
        box.prop(props, "input_image", text="Image")
        box.prop(props, "output_folder", text="Folder")
        box.prop(props, "output_filename", text="Name")
        
        # Generation parameters
        box = layout.box()
        box.label(text="Mesh Parameters", icon='MESH_DATA')
        
        row = box.row()
        row.prop(props, "gen_mesh_width")
        row.prop(props, "gen_focal_length")
        
        row = box.row()
        row.prop(props, "gen_thickness")
        row.prop(props, "gen_resolution")
        
        row = box.row()
        row.prop(props, "gen_verbose")
        
        # Progress display
        if gen_state.is_running or gen_state.progress > 0:
            box = layout.box()
            box.label(text="Progress", icon='TIME')
            
            # Progress bar
            row = box.row()
            row.prop(
                context.scene, "caustic_progress_display",
                text=gen_state.status_text,
                slider=True
            )
            
            # Status text
            box.label(text=f"Status: {gen_state.status_text}")
        
        # Generate button
        layout.separator()
        if gen_state.is_running:
            row = layout.row()
            row.alert = True
            row.operator("caustic.cancel_generation", text="Cancel", icon='CANCEL')
        else:
            layout.operator("caustic.generate_mesh", text="Generate", icon='PLAY')


class CAUSTIC_PT_render_panel(bpy.types.Panel):
    """Panel for scene setup and rendering."""
    bl_label = "Scene Setup & Render"
    bl_idname = "CAUSTIC_PT_render_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Caustic'
    bl_order = 1
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.caustic_props
        
        # File input
        box = layout.box()
        box.label(text="Lens File", icon='MESH_DATA')
        box.prop(props, "obj_path", text="")
        
        # Dimensions
        box = layout.box()
        box.label(text="Dimensions (mm)", icon='ARROW_LEFTRIGHT')
        box.prop(props, "lens_width")
        box.prop(props, "focal_length")
        
        # Light settings
        box = layout.box()
        box.label(text="Light Settings", icon='LIGHT_POINT')
        box.prop(props, "light_distance_multiplier")
        box.prop(props, "light_energy")
        
        # Material settings
        box = layout.box()
        box.label(text="Material", icon='MATERIAL')
        box.prop(props, "glass_ior")
        
        # Render settings
        box = layout.box()
        box.label(text="Render", icon='RENDER_STILL')
        box.prop(props, "render_samples")
        
        # Buttons
        layout.separator()
        layout.operator("caustic.full_setup", text="Full Scene Setup", icon='PLAY')
        
        layout.separator()
        col = layout.column(align=True)
        col.operator("caustic.import_lens", icon='IMPORT')
        col.operator("caustic.setup_light", icon='LIGHT')
        col.operator("caustic.setup_projection_plane", icon='MESH_PLANE')
        col.operator("caustic.setup_render", icon='SCENE')
        
        layout.separator()
        layout.operator("render.render", text="Render", icon='RENDER_STILL')


# Custom property for progress display
def get_progress(self):
    global gen_state
    return gen_state.progress

def set_progress(self, value):
    pass  # Read-only


# Registration
classes = [
    CausticProperties,
    CAUSTIC_OT_generate_mesh,
    CAUSTIC_OT_cancel_generation,
    CAUSTIC_OT_detect_platform,
    CAUSTIC_OT_import_lens,
    CAUSTIC_OT_setup_light,
    CAUSTIC_OT_setup_projection_plane,
    CAUSTIC_OT_setup_render,
    CAUSTIC_OT_full_setup,
    CAUSTIC_PT_generate_panel,
    CAUSTIC_PT_render_panel,
]


def register():
    # Register progress property
    bpy.types.Scene.caustic_progress_display = FloatProperty(
        name="Progress",
        get=get_progress,
        set=set_progress,
        min=0.0,
        max=1.0,
        subtype='FACTOR'
    )
    
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.caustic_props = PointerProperty(type=CausticProperties)
    
    # Print info on registration
    platform_info = get_platform_info()
    print(f"Caustic Lens Renderer add-on registered on {platform_info['system']}")
    if platform_info['is_wsl_available']:
        print(f"  WSL available: {platform_info['wsl_distro']}")
    print("Find it in 3D View sidebar (N) > Caustic tab")


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
    
    if hasattr(bpy.types.Scene, 'caustic_props'):
        del bpy.types.Scene.caustic_props
    if hasattr(bpy.types.Scene, 'caustic_progress_display'):
        del bpy.types.Scene.caustic_progress_display


if __name__ == "__main__":
    register()
