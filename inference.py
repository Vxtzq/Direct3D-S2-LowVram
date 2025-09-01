from direct3d_s2.pipeline import Direct3DS2Pipeline
from torch.cuda.amp import autocast
pipeline = Direct3DS2Pipeline.from_pretrained(
  'wushuang98/Direct3D-S2', 
  subfolder="direct3d-s2-v-1-1"
)
pipeline.to("cuda:0")
with autocast(True):
  mesh = pipeline(
    'assets/0.png', 
    sdf_resolution=1024, # 512 or 1024
    remove_interior=True,
    remesh=True, # Switch to True if you need to reduce the number of triangles.
  )["mesh"]

mesh.export('output.obj')
