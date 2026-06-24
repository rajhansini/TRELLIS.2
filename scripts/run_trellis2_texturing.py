"""
Apply TRELLIS.2 texturing pipeline to a sequence of per-frame images,
using a single fixed mesh, and produce the same folder structure as trellis_glbs/.

Output layout:
    output_dir/
        frame_0001/
            frame_0001.glb       <- TRELLIS.2 textured GLB
            renders/
                front.png
                front_right.png
                right.png
                back.png
                left.png
                front_left.png
            converted/
                mesh.obj
                mesh.mtl
                texture.png
        frame_0002/
        ...

Usage:
    python scripts/run_trellis2_texturing.py \\
        --mesh     /path/to/teapot_homogenized.obj \\
        --frames   /path/to/teapot_lava_kling_premium_front \\
        --output   /path/to/trellis2_glbs \\
        [--model   microsoft/TRELLIS.2-4B] \\
        [--resolution 1024] \\
        [--texture_size 2048] \\
        [--seed 42]
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Camera params — same as trellis_glbs renders
VIEW_NAMES    = ["front", "front_right", "right", "back", "left", "front_left"]
VIEW_AZIMUTHS = [90.0,    45.0,          0.0,     -90.0,  -180.0, -225.0]
ELEVATION_DEG = 0.0
RADIUS        = 1.5
FOV_DEG       = 60.0


def render_views(glb_path: Path, renders_dir: Path, render_size: int = 1024):
    import numpy as np
    import trimesh
    import pyrender
    from PIL import Image

    scene_mesh = trimesh.load(str(glb_path))
    if isinstance(scene_mesh, trimesh.Scene):
        meshes = []
        for name, geom in scene_mesh.geometry.items():
            node_xforms = scene_mesh.graph.get(frame_to=name)
            if node_xforms is not None:
                geom = geom.copy()
                geom.apply_transform(node_xforms[0])
            meshes.append(geom)
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    else:
        mesh = scene_mesh

    # Normalize to [-0.5, 0.5]^3
    verts = np.array(mesh.vertices, dtype=np.float64)
    center = (verts.min(axis=0) + verts.max(axis=0)) / 2
    extent = (verts.max(axis=0) - verts.min(axis=0)).max()
    mesh.vertices = (verts - center) / (extent + 1e-8)

    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.3, 0.3, 0.3])
    scene.add(pr_mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.radians(FOV_DEG))
    light  = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    r = pyrender.OffscreenRenderer(render_size, render_size)
    renders_dir.mkdir(parents=True, exist_ok=True)

    for name, azim_deg in zip(VIEW_NAMES, VIEW_AZIMUTHS):
        azim = np.radians(azim_deg)
        elev = np.radians(ELEVATION_DEG)
        eye  = np.array([
            RADIUS * np.cos(elev) * np.cos(azim),
            RADIUS * np.sin(elev),
            RADIUS * np.cos(elev) * np.sin(azim),
        ])
        z = eye / np.linalg.norm(eye)
        up = np.array([0.0, 1.0, 0.0])
        x  = np.cross(up, z); x /= np.linalg.norm(x)
        y  = np.cross(z, x)
        cam_pose = np.eye(4)
        cam_pose[:3, 0] = x; cam_pose[:3, 1] = y
        cam_pose[:3, 2] = z; cam_pose[:3, 3] = eye

        cn = scene.add(camera, pose=cam_pose)
        ln = scene.add(light,  pose=cam_pose)
        color, _ = r.render(scene)
        scene.remove_node(cn); scene.remove_node(ln)
        Image.fromarray(color).save(str(renders_dir / f"{name}.png"))

    r.delete()


def export_converted(glb_path: Path, conv_dir: Path):
    import trimesh
    conv_dir.mkdir(parents=True, exist_ok=True)
    scene = trimesh.load(str(glb_path), force='scene')
    export = trimesh.exchange.obj.export_obj(
        scene.to_mesh() if hasattr(scene, 'to_mesh') else scene,
        include_texture=True
    )
    (conv_dir / "mesh.obj").write_text(export['obj'])
    if export.get('mtl'):
        (conv_dir / "mesh.mtl").write_text(export['mtl'])
    for fname, data in export.get('textures', {}).items():
        (conv_dir / fname).write_bytes(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh",         required=True,  help="Path to fixed input mesh (OBJ/GLB/PLY)")
    parser.add_argument("--frames",       required=True,  help="Dir with frame_XXXX.png images")
    parser.add_argument("--output",       required=True,  help="Output directory (trellis2_glbs)")
    parser.add_argument("--model",        default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--resolution",   type=int, default=1024, choices=[512, 1024])
    parser.add_argument("--texture_size", type=int, default=2048)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--render_size",  type=int, default=1024)
    parser.add_argument("--guidance_strength", type=float, default=3.0)
    parser.add_argument("--steps",             type=int,   default=12)
    parser.add_argument("--rescale_t",         type=float, default=3.0)
    args = parser.parse_args()

    import torch
    import trimesh
    from PIL import Image
    from trellis2.pipelines import Trellis2TexturingPipeline

    frames_dir = Path(args.frames)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_images = sorted(frames_dir.glob("frame_*.png"))
    if not frame_images:
        raise RuntimeError(f"No frame_*.png found in {frames_dir}")
    print(f"Found {len(frame_images)} frames")

    # Load fixed mesh once
    base_mesh = trimesh.load(str(args.mesh))
    if isinstance(base_mesh, trimesh.Scene):
        base_mesh = base_mesh.to_mesh()
    print(f"Loaded mesh: {args.mesh}  ({len(base_mesh.vertices)} verts, {len(base_mesh.faces)} faces)")

    print(f"Loading pipeline from {args.model} ...")
    pipeline = Trellis2TexturingPipeline.from_pretrained(
        args.model, config_file="texturing_pipeline.json"
    )
    pipeline.cuda()
    print("Pipeline ready.\n")

    sampler_params = {
        "steps": args.steps,
        "guidance_strength": args.guidance_strength,
        "guidance_rescale": 0.0,
        "rescale_t": args.rescale_t,
    }

    for img_path in frame_images:
        frame_name = img_path.stem          # e.g. "frame_0001"
        frame_dir  = output_dir / frame_name
        out_glb    = frame_dir / f"{frame_name}.glb"

        if out_glb.exists():
            print(f"  {frame_name}: already done, skipping")
            continue

        print(f"  {frame_name}: {img_path.name}", flush=True)
        frame_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(str(img_path)).convert("RGBA")

        # Fresh copy of mesh each frame (pipeline mutates vertices internally)
        mesh = trimesh.Trimesh(
            vertices=base_mesh.vertices.copy(),
            faces=base_mesh.faces.copy(),
            process=False,
        )
        if hasattr(base_mesh, 'visual') and hasattr(base_mesh.visual, 'uv') and base_mesh.visual.uv is not None:
            mesh.visual = trimesh.visual.TextureVisuals(uv=base_mesh.visual.uv.copy(), material=base_mesh.visual.material)

        textured = pipeline.run(
            mesh, image,
            seed=args.seed,
            preprocess_image=True,
            resolution=args.resolution,
            texture_size=args.texture_size,
            tex_slat_sampler_params=sampler_params,
        )
        textured.export(str(out_glb), extension_webp=True)
        print(f"    GLB saved", flush=True)

        try:
            render_views(out_glb, frame_dir / "renders", args.render_size)
            print(f"    renders saved", flush=True)
        except Exception as e:
            print(f"    renders skipped: {e}", flush=True)

        try:
            export_converted(out_glb, frame_dir / "converted")
            print(f"    converted saved", flush=True)
        except Exception as e:
            print(f"    converted skipped: {e}", flush=True)

        torch.cuda.empty_cache()

    total = sum(1 for p in output_dir.glob("*/frame_*.glb"))
    print(f"\nDone — {total}/{len(frame_images)} frames in {output_dir}")


if __name__ == "__main__":
    main()
