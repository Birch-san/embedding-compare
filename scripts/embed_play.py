import data
import torch
import data
from models import imagebind_model
from models.imagebind_model import ModalityType, ImageBindModel
from os import listdir, makedirs
from os.path import join
from pathlib import Path
import fnmatch
from typing import Callable, List
from src.tabulation import tabulate_similarity

# relative to current working directory, i.e. repository root of embedding-compare
img_bind_dir = 'lib/ImageBind'

data.BPE_PATH = join(img_bind_dir, data.BPE_PATH)
img_bind_assets_dir = join(img_bind_dir, '.assets')
assets_dir = 'assets'

text_list=['Reimu', 'Flandre', 'Kochiya Sanae', 'Patchouli Knowlege', 'Rem', 'Saber', 'Matou Sakura', 'Youmu', 'anime girl', 'illustration', 'national anthem', 'bossa nova', 'chiptune']
image_stems = ['reimu1', 'flandre1', 'sanae1', 'patchouli0', 'rem0', 'saber1', 'sakura1', 'youmu4']
image_paths=[join(assets_dir, f'{asset}.jpg') for asset in image_stems]
audio_stems=['reimu', 'flandre', 'sanae', 'patchouli', 'rem1', 'saber', 'sakura-saber', 'youmu', 'british-anthem']
audio_paths=[join(assets_dir, f'{asset}.wav') for asset in audio_stems]

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.set_printoptions(sci_mode=False)

# Load data
inputs = {
  ModalityType.TEXT: data.load_and_transform_text(text_list, device),
  ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
  ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

# Instantiate model
model: ImageBindModel = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

with torch.no_grad():
  embeddings = model(inputs)


# print(
#   'Vision x Text: ',
#   torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
# )
print(
  "Vision x Text: \n",
  tabulate_similarity(
    mode0=embeddings[ModalityType.VISION],
    mode1=embeddings[ModalityType.TEXT],
    mode0_modality=ModalityType.VISION,
    mode0_labels=image_stems,
    mode1_labels=text_list,
  )
)
print(
  "Text x Vision: \n",
  tabulate_similarity(
    mode0=embeddings[ModalityType.TEXT],
    mode1=embeddings[ModalityType.VISION],
    mode0_modality=ModalityType.TEXT,
    mode0_labels=text_list,
    mode1_labels=image_stems,
  )
)
# print(
#   'Audio x Text: ',
#   torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1),
# )
print(
  "Audio x Text: \n",
  tabulate_similarity(
    mode0=embeddings[ModalityType.AUDIO],
    mode1=embeddings[ModalityType.TEXT],
    mode0_modality=ModalityType.AUDIO,
    mode0_labels=audio_stems,
    mode1_labels=text_list,
  )
)
print(
  "Text x Audio: \n",
  tabulate_similarity(
    mode0=embeddings[ModalityType.TEXT],
    mode1=embeddings[ModalityType.AUDIO],
    mode0_modality=ModalityType.TEXT,
    mode0_labels=text_list,
    mode1_labels=audio_stems,
  )
)
# print(
#   'Vision x Audio: ',
#   torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1),
# )
print(
  "Audio x Vision: \n",
  tabulate_similarity(
    mode0=embeddings[ModalityType.AUDIO],
    mode1=embeddings[ModalityType.VISION],
    mode0_modality=ModalityType.AUDIO,
    mode0_labels=audio_stems,
    mode1_labels=image_stems,
  )
)
print(
  "Vision x Audio: \n",
  tabulate_similarity(
    mode0=embeddings[ModalityType.VISION],
    mode1=embeddings[ModalityType.AUDIO],
    mode0_modality=ModalityType.VISION,
    mode0_labels=image_stems,
    mode1_labels=audio_stems,
  )
)

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