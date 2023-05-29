import data
import torch
import data
from models import imagebind_model
from models.imagebind_model import ModalityType, ImageBindModel
from os import listdir, makedirs
from os.path import join
from pathlib import Path
import fnmatch
from dominate import document
from typing import Callable, List, Dict, Optional
from numpy.typing import NDArray
from itertools import permutations
from shutil import copyfile
from torch import Tensor
from src.markup import markup_similarities, out_assets_dir_rel
from src.tabulation import tabulate_similarity
from src.similarity import get_similarity

# relative to current working directory, i.e. repository root of embedding-compare
img_bind_dir = 'lib/ImageBind'

data.BPE_PATH = join(img_bind_dir, data.BPE_PATH)
img_bind_assets_dir = join(img_bind_dir, '.assets')
assets_dir = 'assets'
html_assets_dir = 'html_assets'

text_list=['Reimu', 'Flandre', 'Kochiya Sanae', 'Patchouli Knowlege', 'Rem', 'Saber', 'Matou Sakura', 'Youmu', 'anime girl', 'illustration', 'national anthem', 'bossa nova', 'chiptune']
image_stems = ['reimu1', 'flandre1', 'sanae1', 'patchouli0', 'rem0', 'saber1', 'sakura1', 'youmu4']
image_paths_in=[join(assets_dir, f'{asset}.jpg') for asset in image_stems]
image_paths_out=[join(out_assets_dir_rel, f'{asset}.jpg') for asset in image_stems]
audio_stems=['reimu', 'flandre', 'sanae', 'patchouli', 'rem1', 'saber', 'sakura-saber', 'youmu', 'british-anthem']
audio_paths_in=[join(assets_dir, f'{asset}.wav') for asset in audio_stems]
audio_paths_out=[join(out_assets_dir_rel, f'{asset}.wav') for asset in audio_stems]

modality_names: Dict[ModalityType, List[str]] = {
  ModalityType.VISION: image_stems,
  ModalityType.TEXT: text_list,
  ModalityType.AUDIO: audio_stems,
}
modality_asset_references: Dict[ModalityType, Optional[List[str]]] = {
  ModalityType.VISION: image_paths_out,
  ModalityType.TEXT: None,
  ModalityType.AUDIO: audio_paths_out,
}

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

torch.set_printoptions(sci_mode=False)

# Load data
modalities_with_data: List[ModalityType] = []
inputs: Dict[ModalityType, Tensor] = {}
if text_list:
  inputs[ModalityType.TEXT] = data.load_and_transform_text(text_list, device)
  modalities_with_data += [ModalityType.TEXT]
if image_paths_in:
  inputs[ModalityType.VISION] = data.load_and_transform_vision_data(image_paths_in, device)
  modalities_with_data += [ModalityType.VISION]
if audio_paths_in:
  inputs[ModalityType.AUDIO] = data.load_and_transform_audio_data(audio_paths_in, device)
  modalities_with_data += [ModalityType.AUDIO]

# Instantiate model
model: ImageBindModel = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

with torch.no_grad():
  embeddings = model(inputs)

similarity_dicts: Dict[ModalityType, Dict[ModalityType, NDArray]] = {
  ModalityType.VISION: {},
  ModalityType.TEXT: {},
  ModalityType.AUDIO: {},
}

for source, target in permutations(modalities_with_data, 2):
  similarity_dicts[source][target] = get_similarity(
    embeddings[source],
    embeddings[target],
  ).cpu().numpy()

# yes I know these loops can be combined into the one above, but I prefer
# to handle each type of output in isolation, to simplify breakpoint-debugging
for source_modality, target_modalities in similarity_dicts.items():
  for target_modality, similarities in target_modalities.items():
    print(
      f'{source_modality} x {target_modality}: \n',
      tabulate_similarity(
        similarity=similarity_dicts[source_modality][target_modality],
        mode0_modality=source_modality,
        mode0_names=modality_names[source_modality],
        mode1_names=modality_names[target_modality],
      )
    )

out_root = 'out'
makedirs(out_root, exist_ok=True)

get_out_ix: Callable[[str], int] = lambda stem: int(stem.split('_', maxsplit=1)[0])
out_dirs_unsorted: List[str] = fnmatch.filter(listdir(out_root), f'*_out')
out_keyer: Callable[[str], int] = lambda fname: get_out_ix(Path(fname).stem)
out_dirs: List[str] = [join(out_root, out_dir) for out_dir in sorted(out_dirs_unsorted, key=out_keyer)]
next_ix = get_out_ix(Path(out_dirs[-1]).stem)+1 if out_dirs else 0
out_dir: str = join(out_root, f'{next_ix:03d}_out')
makedirs(out_dir)

print(f'Created output directory: {out_dir}')

out_assets_dir_qual = join(out_dir, out_assets_dir_rel)
makedirs(out_assets_dir_qual, exist_ok=True)
for in_path in [*image_paths_in, *audio_paths_in]:
  copyfile(in_path, join(out_assets_dir_qual, Path(in_path).name))
copyfile(join(html_assets_dir, 'style.css'), join(out_dir, 'style.css'))
# copyfile(join(html_assets_dir, 'interaction.mjs'), join(out_dir, 'interaction.mjs'))

for source_modality, target_modalities in similarity_dicts.items():
  for target_modality, similarities in target_modalities.items():
    out_file = join(out_dir, f'{source_modality}_{target_modality}_similarity.html')
    doc: document = markup_similarities(
      similarity=similarities,
      mode0_modality=source_modality,
      mode1_modality=target_modality,
      mode0_names=modality_names[source_modality],
      mode0_asset_references=modality_asset_references[source_modality],
      mode1_names=modality_names[target_modality],
      mode1_asset_references=modality_asset_references[target_modality],
    )
    markup: str = str(doc)

    with open(out_file, 'w') as f:
      f.write(markup)