from torch import FloatTensor, softmax
from models.imagebind_model import ModalityType
from tabulate import tabulate
from typing import List, Dict
from numpy.typing import NDArray

ansi_purple = '\033[95m'
ansi_reset = '\033[0m'

def format_float(num: float) -> str:
  return f'{num:.2f}'

def highlight(nominal: str, colour=True) -> str:
  return f'{ansi_purple}{nominal}{ansi_reset}' if colour else f'*{nominal}*'

def format_row(
  row: NDArray,
  colour = True,
) -> List[str]:
  return [highlight(format_float(num), colour=colour) if num == row.max() else format_float(num) for num in row]

modality_to_sample_name: Dict[ModalityType, str] = {
  ModalityType.TEXT: 'label',
  ModalityType.VISION: 'image',
  ModalityType.AUDIO: 'sound',
}

def tabulate_similarity(
  similarity: NDArray,
  mode0_modality: ModalityType,
  mode0_labels: str,
  mode1_labels: str,
  colour = True,
) -> str:
  """
  Visualizes similarity, where similarity is defined as:
    softmax(mode0 @ mode1.T, dim=-1)
  In other words:
  for each item in mode0 (rows),
    how similar are the elements in mode1 (columns)?
Vision x Text:
image         Reimu    Marisa
----------  -------  ---------
reimu.jpg      0.98       0.02
marisa.jpg     0             1
  """
  return tabulate(
    [
      [stem, *format_row(row, colour=colour)] for stem, row in zip(
        mode0_labels,
        similarity,
      )
    ],
    headers=[modality_to_sample_name[mode0_modality], *mode1_labels],
  )