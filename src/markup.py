import dominate
from dominate.tags import *
from numpy.typing import NDArray
from models.imagebind_model import ModalityType
from typing import Dict, List, Optional
from pathlib import Path
from contextlib import nullcontext

out_assets_dir_rel='assets'

def format_float(num: float) -> str:
  return f'{num:.2f}'

modality_to_sample_name: Dict[ModalityType, str] = {
  ModalityType.TEXT: 'Label',
  ModalityType.VISION: 'Image',
  ModalityType.AUDIO: 'Sound',
}

def markup_sample(
  modality: ModalityType,
  name: str,
  asset_reference: str,
  head: bool
) -> None:
  cls = th if head else td
  match modality:
    case ModalityType.VISION:
      with cls():
        with a(href=asset_reference, target='_blank'):
          img(src=asset_reference, width='50px', title=name)
    case ModalityType.TEXT:
      with cls(cls='target-label' if head else 'subject-label'):
        div(name, cls='label-text')
    case ModalityType.AUDIO:
      with cls():
        with audio(controls='controls', title=name):
          source(src=asset_reference, type='audio/wav')
    case _:
      raise ValueError(f'Not implemented ModalityType: {modality}')

def markup_similarities(
  similarity: NDArray,
  mode0_modality: ModalityType,
  mode1_modality: ModalityType,
  mode0_names: List[str],
  mode0_asset_references: Optional[List[str]],
  mode1_names: List[str],
  mode1_asset_references: Optional[List[str]],
) -> dominate.document:
  doc = dominate.document(title=f'{mode0_modality} x {mode1_modality} similarity')

  with doc.head:
    link(rel='stylesheet', href='style.css')
    # script(type='module', src='interaction.mjs')

  with doc:
    p(f'{mode0_modality} x {mode1_modality} similarity:')
    with table():
      with thead():
        with tr(cls='subject-target-head'):
          th('Subject', colspan=1 if mode0_asset_references is None else 2, cls='subjects')
          th('Similarity to Target', colspan=len(mode1_names), cls='targets')
        with tr(cls='subject-target-detail'):
          rowspan = 1 if mode1_asset_references is None else 2
          # if mode1_asset_references is not None:
          #   for asset_reference in mode1_asset_references:
          #     th(Path(asset_reference).stem)
          # context = nullcontext() if mode1_asset_references is None else tr
          if mode1_asset_references is None:
            if mode0_asset_references is not None:
              th('Ref', cls='subject-ref')
            th(modality_to_sample_name[mode0_modality], cls='subject-category'),
            for name, asset_reference in zip(
              mode1_names,
              [None]*len(mode1_names) if mode1_asset_references is None else mode1_asset_references,
            ):
              markup_sample(
                modality=mode1_modality,
                name=name,
                asset_reference=asset_reference,
                head=True,
              )
          else:
            with tr():
              if mode0_asset_references is not None:
                th('Ref', cls='subject-ref', rowspan=2)
              th(modality_to_sample_name[mode0_modality], cls='subject-category', rowspan=2),
              for asset_reference in mode1_asset_references:
                th(Path(asset_reference).stem, cls='target-ref')
            with tr():
              for name, asset_reference in zip(
                mode1_names,
                [None]*len(mode1_names) if mode1_asset_references is None else mode1_asset_references,
              ):
                markup_sample(
                  modality=mode1_modality,
                  name=name,
                  asset_reference=asset_reference,
                  head=True,
                )

      with tbody():
        for name, asset_reference, row in zip(
          mode0_names,
          [None]*len(mode0_names) if mode0_asset_references is None else mode0_asset_references,
          similarity,
        ):
          with tr():
            if mode0_asset_references is not None:
              td(Path(asset_reference).stem, cls='subject-ref')
            markup_sample(
              modality=mode0_modality,
              name=name,
              asset_reference=asset_reference,
              head=False,
            )
            for num in row:
              td(format_float(num), cls=f"similarity-number {'similarity-beeg' if num == row.max() else ''} {'similarity-none' if num < 0.01 else ''}")

  return doc