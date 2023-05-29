from os import listdir, makedirs
from os.path import join
from typing import List, Callable, Dict, Optional
import fnmatch
from pathlib import Path
from numpy import array
from numpy.typing import NDArray
from src.markup import markup_similarities, out_assets_dir_rel
from src.similarity_fixtures import similarity_dicts
from dominate import document
from models.imagebind_model import ModalityType
from shutil import copyfile

html_assets_dir = 'html_assets'
in_assets_dir = 'assets'

text_list=['Reimu', 'Flandre', 'Kochiya Sanae', 'Patchouli Knowlege', 'Rem', 'Saber', 'Matou Sakura', 'Youmu', 'anime girl', 'illustration', 'national anthem', 'bossa nova', 'chiptune']
image_stems = ['reimu1', 'flandre1', 'sanae1', 'patchouli0', 'rem0', 'saber1', 'sakura1', 'youmu4']
image_paths_in=[join(in_assets_dir, f'{asset}.jpg') for asset in image_stems]
image_paths_out=[join(out_assets_dir_rel, f'{asset}.jpg') for asset in image_stems]
audio_stems=['reimu', 'flandre', 'sanae', 'patchouli', 'rem1', 'saber', 'sakura-saber', 'youmu', 'british-anthem']
audio_paths_in=[join(in_assets_dir, f'{asset}.wav') for asset in audio_stems]
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

out_root = 'out'
makedirs(out_root, exist_ok=True)

# get_out_ix: Callable[[str], int] = lambda stem: int(stem.split('_', maxsplit=1)[0])
# out_dirs_unsorted: List[str] = fnmatch.filter(listdir(out_root), f'*_out')
# out_keyer: Callable[[str], int] = lambda fname: get_out_ix(Path(fname).stem)
# out_dirs: List[str] = [join(out_root, out_dir) for out_dir in sorted(out_dirs_unsorted, key=out_keyer)]
# next_ix = get_out_ix(Path(out_dirs[-1]).stem)+1 if out_dirs else 0
# out_dir: str = join(out_root, f'{next_ix:03d}_out')
# makedirs(out_dir)

# print(f'Created output directory: {out_dir}')

out_dir = join(out_root, '000_out')
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