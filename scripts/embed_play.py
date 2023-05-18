import data
import torch
import data
from models import imagebind_model
from models.imagebind_model import ModalityType
from os import listdir, makedirs
from os.path import join
from pathlib import Path
import platform
import fnmatch
from typing import Callable, List

# relative to current working directory, i.e. repository root of embedding-compare
img_bind_dir = 'lib/ImageBind'

data.BPE_PATH = join(img_bind_dir, data.BPE_PATH)
img_bind_assets_dir = join(img_bind_dir, '.assets')
assets_dir = 'assets'

text_list=['A dog.', 'A car', 'A bird']
image_paths=[join(img_bind_assets_dir, asset) for asset in ['dog_image.jpg', 'car_image.jpg', 'bird_image.jpg']]
audio_paths=[join(img_bind_assets_dir, asset) for asset in ['dog_audio.wav', 'car_audio.wav', 'bird_audio.wav']]

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load data
inputs = {
  ModalityType.TEXT: data.load_and_transform_text(text_list, device),
  ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
  ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

torch.compile(model, mode='max-autotune')

with torch.no_grad():
    embeddings = model(inputs)

print(platform.python_version())
print(torch.__version__)


get_out_ix: Callable[[str], int] = lambda stem: int(stem.split('_', maxsplit=1)[0])
out_root = 'out'
makedirs(out_root, exist_ok=True)

out_dirs_unsorted: List[str] = fnmatch.filter(listdir(out_root), f'*_out')
out_keyer: Callable[[str], int] = lambda fname: get_out_ix(Path(fname).stem)
out_dirs: List[str] = [join(out_root, out_dir) for out_dir in sorted(out_dirs_unsorted, key=out_keyer)]
next_ix = get_out_ix(Path(out_dirs[-1]).stem)+1 if out_dirs else 0
out_dir: str = join(out_root, f'{next_ix:03d}_out')
makedirs(out_dir)

print(f'Created output directory: {out_dir}')

# Expected output:
#
# Vision x Text:
# tensor([[9.9761e-01, 2.3694e-03, 1.8612e-05],
#         [3.3836e-05, 9.9994e-01, 2.4118e-05],
#         [4.7997e-05, 1.3496e-02, 9.8646e-01]])
#
# Audio x Text:
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])
#
# Vision x Audio:
# tensor([[0.8070, 0.1088, 0.0842],
#         [0.1036, 0.7884, 0.1079],
#         [0.0018, 0.0022, 0.9960]])