
import torch
import trimesh
import datetime
import argparse
import numpy as np 
from torchvision import transforms
from direct3d_s2.utils.rembg import BiRefNet
from direct3d_s2.pipeline import Direct3DS2Pipeline
from direct3d_s2.utils.fill_hole import postprocess_mesh

import os
from PIL import Image
from typing import Any

import gradio as gr
from gradio.themes.utils import colors, fonts, sizes

# -----------------------------------------------------------------------------
#  THEME  ▸  a soft glass-like dark theme with a vibrant primary accent
# -----------------------------------------------------------------------------
class Glass(gr.themes.Soft):
    def __init__(self):
        super().__init__(
            primary_hue=colors.emerald,
            secondary_hue=colors.indigo,
            neutral_hue=colors.zinc,
            text_size=sizes.text_md,
            spacing_size=sizes.spacing_md,
            radius_size=sizes.radius_lg,
            font=fonts.GoogleFont("Inter"),
        )

    def style(self):
        super().style()
        self.set(
            background_fill="var(--neutral-950)",
            border_color_primary="rgba(255,255,255,.12)",
            border_width="1px",
            shadow_drop="0 10px 38px -10px rgba(0,0,0,.65)",
            shadow_drop_lg="0 10px 38px -10px rgba(0,0,0,.65)",
        )
        return self

def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")

# -----------------------------------------------------------------------------
#  PLACEHOLDER BACK-END HOOKS  ▸  replace with your real logic
# -----------------------------------------------------------------------------
def image2mesh(
    image: Any, 
    resolution: str = '1024', 
    simplify: bool = True,
    simplify_ratio: float = 0.95, 
    output_path: str = 'outputs/web'
):
    
    torch.cuda.empty_cache()
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    uid = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    image.save(os.path.join(output_path, uid + '.png'))

    mesh = pipe(
        image, 
        sdf_resolution=int(resolution), 
        mc_threshold=0.2,
        remesh=simplify,
        simplify_ratio=simplify_ratio,
    )["mesh"]

    mesh_path = os.path.join(output_path, f'{uid}.obj')
    mesh.export(
        mesh_path,
        include_normals=True,
    )
    torch.cuda.empty_cache()
    
    return mesh_path

# -----------------------------------------------------------------------------
#  UI LAYOUT  ▸  minimal glassmorphism, keyboard-first workflow
# -----------------------------------------------------------------------------

pipe = Direct3DS2Pipeline.from_pretrained('wushuang98/Direct3D-S2', subfolder="direct3d-s2-v-1-1")
pipe.to("cuda:0")

with gr.Blocks(theme=Glass(), css="""
:root { --header-height:64px }
body { background:linear-gradient(215deg,#101113 0%,#0b0c0d 60%,#0d1014 100%) }
#header { height:var(--header-height);display:flex;align-items:center;justify-content:space-between;padding:0 1.5rem;backdrop-filter:blur(18px);background:rgba(17,17,17,.65);border-bottom:1px solid rgba(255,255,255,.08);position:sticky;top:0;z-index:999 }
#header a { color:white;font-weight:500;text-decoration:none;margin-right:1.25rem;font-size:.925rem }
#hero-title { font-size:1.35rem;font-weight:600;color:white;white-space:nowrap }
#footer { text-align:center;font-size:.8rem;color:rgba(255,255,255,.55);margin-top:1.5rem }
#mesh_viewport { aspect-ratio:1/1;width:100%;display:flex;align-items:center;justify-content:center;border:1px dashed rgba(255,255,255,.12);border-radius:12px;background:rgba(255,255,255,.03); }
.gallery-item img { border-radius:10px }
#examples_gallery { height:100%;flex:1;display:flex;flex-direction:column; }
#examples_gallery img { width:800px;}
#show_image img { height:260px;display:flex;align-items:center;justify-content:center; }
#examples { height:100%;flex:1; }
""") as demo:

    # ▸ custom sticky header
    with gr.Row(elem_id="header", variant="panel"):
        gr.Markdown("<span id='hero-title'>Direct3D-S2 Studio</span>", elem_id="hero-title")
        gr.Markdown(
            """<span>
            </span>""",
            elem_id="nav-links",
        )

    # ▸ main workspace
    with gr.Row(equal_height=True):
        # ---------- Controls ----------
        with gr.Column(scale=3):
            gr.Markdown("### Input", elem_classes="subtitle")
            image_input = gr.Image(
                label="Image Input",
                image_mode="RGBA",
                sources="upload",
                type="pil",
                height=260, 
                elem_id="show_image",
            )
            processed_image = gr.Image(
                label="Processed Image",
                image_mode="RGBA",
                type="pil",
                interactive=False,
                height=260, 
                elem_id="show_image",
            )
            with gr.Accordion("Advanced Options", open=True):
                use_alpha = gr.Checkbox(label="Use Alpha Channel", value=False)
                resolution = gr.Radio(choices=["512", "1024"], label="SDF Resolution", value="1024")
                simplify = gr.Checkbox(label="Simplify Mesh", value=True)
                reduce_ratio = gr.Slider(0.1, 0.95, step=0.05, value=0.95, label="Faces Reduction Ratio")
                
            gen_btn = gr.Button("Generate 3D ✨", variant="primary", interactive=True)

        # ---------- Viewport ----------
        with gr.Column(scale=6):
            gr.Markdown("### Model Viewer", elem_classes="subtitle")
            # mesh_html = gr.HTML("<div id='mesh_viewport'>🌀 No mesh yet</div>")
            output_model_obj = gr.Model3D(
                label="Output Model (OBJ Format)",
                camera_position=(90.0, 90.0, 3.5),
                interactive=False,
                elem_id="mesh_viewport",
            )

        # ---------- Gallery / Examples ----------
        with gr.Column(scale=3):
            gr.Markdown("### Examples", elem_classes="subtitle")
            with gr.Tabs(selected='tab_img_gallery') as gallery:
                with gr.Tab('Image to 3D Gallery', id='tab_img_gallery') as tab_gi:
                    with gr.Row():
                        gr.Examples(
                            examples=[os.path.join("assets/test", i) for i in os.listdir("assets/test")], 
                            inputs=[image_input],
                            label=None, 
                            examples_per_page=24
                        )

    # ▸ callbacks
    outputs = [output_model_obj]
    rmbg = BiRefNet(device="cuda:0")

    gen_btn.click(
        fn=check_input_image, 
        inputs=[image_input]
    ).success(
        fn=rmbg.run, 
        inputs=[image_input, use_alpha],
        outputs=[processed_image]
    ).success(
        fn=image2mesh, 
        inputs=[processed_image, resolution, simplify, reduce_ratio],
        outputs=outputs, 
        api_name="generate_img2obj"
    )

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached_dir", type=str, default="outputs/web")
    args = parser.parse_args()
    
    demo.queue().launch(share=True, allowed_paths=[args.cached_dir], server_port=7860)
