from os import listdir, makedirs
from os.path import join
from typing import List, Callable
import fnmatch
from pathlib import Path
from numpy import array
from numpy.typing import NDArray
from src.markup import markup_similarities, out_assets_dir_rel
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

# Vision x Text: 
#  image         Reimu    Flandre    Kochiya Sanae    Patchouli Knowlege    Rem    Saber    Matou Sakura    Youmu    anime girl    illustration    national anthem    bossa nova    chiptune
# ----------  -------  ---------  ---------------  --------------------  -----  -------  --------------  -------  ------------  --------------  -----------------  ------------  ----------
# reimu1         0.98          0             0                     0      0           0            0        0             0.02               0                  0             0           0
# flandre1       0             1             0                     0      0           0            0        0             0                  0                  0             0           0
# sanae1         0             0             0.88                  0      0           0            0        0.08          0.03               0                  0             0           0
# patchouli0     0.4           0             0                     0.46   0           0            0        0             0.14               0                  0             0           0
# rem0           0             0             0.02                  0      0.97        0            0        0.01          0.01               0                  0             0           0
# saber1         0             0             0                     0      0           1            0        0             0                  0                  0             0           0
# sakura1        0.03          0             0.23                  0.5    0           0            0.21     0             0.03               0                  0             0           0
# youmu4         0             0             0.01                  0      0           0            0        0.99          0                  0                  0             0           0
#
# numpy.set_printoptions(suppress=True, precision=2, linewidth=200)
# torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1).cpu().numpy()
similarities: NDArray = array([
  [0.98, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.02, 0.  , 0.  , 0.  , 0.  ],
  [0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
  [0.  , 0.  , 0.88, 0.  , 0.  , 0.  , 0.  , 0.08, 0.03, 0.  , 0.  , 0.  , 0.  ],
  [0.4 , 0.  , 0.  , 0.46, 0.  , 0.  , 0.  , 0.  , 0.14, 0.  , 0.  , 0.  , 0.  ],
  [0.  , 0.  , 0.02, 0.  , 0.97, 0.  , 0.  , 0.01, 0.01, 0.  , 0.  , 0.  , 0.  ],
  [0.  , 0.  , 0.  , 0.  , 0.  , 1.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
  [0.03, 0.  , 0.23, 0.5 , 0.  , 0.  , 0.21, 0.  , 0.03, 0.  , 0.  , 0.  , 0.  ],
  [0.  , 0.  , 0.01, 0.  , 0.  , 0.  , 0.  , 0.99, 0.  , 0.  , 0.  , 0.  , 0.  ]
])

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
out_file = join(out_dir, 'vision_text_similarity.html')
for img_path in image_paths_in:
  copyfile(img_path, join(out_assets_dir_qual, Path(img_path).name))
copyfile(join(html_assets_dir, 'style.css'), join(out_dir, 'style.css'))
# copyfile(join(html_assets_dir, 'interaction.mjs'), join(out_dir, 'interaction.mjs'))

doc: document = markup_similarities(
  similarity=similarities,
  mode0_modality=ModalityType.VISION,
  mode1_modality=ModalityType.TEXT,
  mode0_names=image_stems,
  mode0_asset_references=image_paths_out,
  mode1_names=text_list,
  mode1_asset_references=None,
)
print(doc)

markup: str = str(doc)

with open(out_file, 'w') as f:
  f.write(markup)