import dominate
from dominate.tags import *
from numpy.typing import NDArray
from models.imagebind_model import ModalityType
from typing import Dict
from os.path import join

def format_float(num: float) -> str:
  return f'{num:.2f}'

modality_to_sample_name: Dict[ModalityType, str] = {
  ModalityType.TEXT: 'Label',
  ModalityType.VISION: 'Image',
  ModalityType.AUDIO: 'Sound',
}

def markup_sample(
  modality: ModalityType,
  asset_name: str,
  head = False,
) -> None:
  cls = th if head else td
  match modality:
    case ModalityType.VISION:
      with cls():
        with figure():
          img(src=join('assets', f'{asset_name}.jpg'), width='50px')
          figcaption(asset_name)
    case ModalityType.TEXT | ModalityType.AUDIO:
      cls(asset_name)
    case _:
      raise ValueError(f'Not implemented ModalityType: {modality}')

def markup_similarities(
  similarity: NDArray,
  mode0_modality: ModalityType,
  mode1_modality: ModalityType,
  mode0_labels: str,
  mode1_labels: str,
) -> dominate.document:
  doc = dominate.document(title=f'{mode0_modality} x {mode1_modality} similarity')

  with doc.head:
    link(rel='stylesheet', href='style.css')

  with doc:
    p(f'{mode0_modality} x {mode1_modality} similarity:')
    with table():
      with thead():
        with tr():
          th('Subject')
          th('Target', colspan=len(mode1_labels))
        with tr():
          th(modality_to_sample_name[mode0_modality]),
          for label in mode1_labels:
            th(label)
      with tbody():
        for stem, row in zip(mode0_labels, similarity):
          with tr():
            markup_sample(
              modality=mode0_modality,
              asset_name=stem,
              head=False,
            )
            for num in row:
              td(format_float(num), cls='similarity-beeg' if num == row.max() else None)

  return doc