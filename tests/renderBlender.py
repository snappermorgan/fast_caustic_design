import bpy
import sys
import os

def process_and_render(base_blend_path, input_obj_path, output_img_path):
    # 1. Load the pre-existing template scene
    # This ensures your Lights, Camera, and "Screen" plane are exactly as you want them
    try:
        bpy.ops.wm.open_mainfile(filepath=base_blend_path)
    except Exception as e:
        print(f"Error opening base scene: {e}")
        return
    
    # 2. Import the generated OBJ (Blender 4.0+ syntax)
    bpy.ops.wm.obj_import(filepath=input_obj_path)
    lens_obj = bpy.context.selected_objects[0]  # Get the imported mesh
    
    # 3. Apply Shade Smooth
    bpy.context.view_layer.objects.active = lens_obj
    bpy.ops.object.shade_smooth()
    
    # 4. Position and Setup
    # Place lens above the floor (floor is at Z=-0.08, lens needs clearance)
    lens_obj.location = (0.0, -1.55, 0.5) 
    
    # Enable Caustics on the Object (Blender version compatibility)
    if bpy.app.version >= (5, 0, 0):
        lens_obj.cycles.is_caustics_caster = True
    elif bpy.app.version >= (4, 0, 0):
        lens_obj.is_caustics_caster = True
    else:
        lens_obj.cycles.cast_shadow_caustics = True
    
    # 5. Create and Assign Material
    mat = bpy.data.materials.new(name="CausticGlass")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # Clear default nodes
    nodes.clear()
    
    # Create Principled BSDF
    shader = nodes.new(type='ShaderNodeBsdfPrincipled')
    shader.inputs['Roughness'].default_value = 0.0
    shader.inputs['Transmission Weight'].default_value = 1.0
    shader.inputs['IOR'].default_value = 1.5 # Standard glass IOR
    
    # Create Output node
    output = nodes.new(type='ShaderNodeOutputMaterial')
    
    # Link them
    links = mat.node_tree.links
    links.new(shader.outputs['BSDF'], output.inputs['Surface'])
    
    # Assign material to object
    if lens_obj.data.materials:
        lens_obj.data.materials[0] = mat
    else:
        lens_obj.data.materials.append(mat)

    # 6. Render Settings
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    
    # GPU Compute - use CUDA for WSL2 compatibility
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'
    prefs.get_devices()
    
    # Debug: print available devices
    print("Available compute devices:")
    gpu_found = False
    for device in prefs.devices:
        print(f"  {device.name}: type={device.type}, use={device.use}")
        if device.type == 'CUDA':
            device.use = True
            gpu_found = True
        else:
            device.use = False
    
    if not gpu_found:
        print("WARNING: No CUDA GPU found, falling back to CPU")
        scene.cycles.device = 'CPU'
    
    # Reduce sample count (caustics need fewer samples with denoiser)
    scene.cycles.samples = 256
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01
    
    # Enable denoiser - use OpenImageDenoise (works everywhere)
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    
    # Reduce light bounces (caustics mainly need transmission)
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 8
    scene.cycles.transparent_max_bounces = 8
    
    # Reduce tile size for GPU (Blender 3.0+ uses dynamic tiles, but this helps)
    scene.cycles.tile_size = 256
    
    # Disable motion blur and DOF if not needed
    scene.render.use_motion_blur = False
    if scene.camera and scene.camera.data.dof.use_dof:
        scene.camera.data.dof.use_dof = False

    # Lower resolution for quick tests (comment out for final)
    # scene.render.resolution_percentage = 50
    
    # Set Output path
    scene.render.filepath = output_img_path
    
    # 7. Render and Save
    print("Starting Render...")
    bpy.ops.render.render(write_still=True)
    print(f"Render saved to {output_img_path}")

# Argument Parsing
# Blender passes its own arguments, so we look for a double dash '--' separator
if "--" in sys.argv:
    args = sys.argv[sys.argv.index("--") + 1:]
    if len(args) < 3:
        print("Usage: blender -b ... -- <base_blend> <input_obj> <output_png>")
    else:
        process_and_render(args[0], args[1], args[2])