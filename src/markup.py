import dominate
from dominate.tags import *
from numpy.typing import NDArray
from models.imagebind_model import ModalityType

def markup_similarities(
  similarity: NDArray,
  mode0_modality: ModalityType,
  mode0_labels: str,
  mode1_labels: str,
) -> dominate.document:
  doc = dominate.document(title='Dominate your HTML')

  with doc.head:
    link(rel='stylesheet', href='style.css')
    script(type='text/javascript', src='script.js')

  with doc:
    with div(id='header').add(ol()):
      for i in ['home', 'about', 'contact']:
        li(a(i.title(), href='/%s.html' % i))

      with div():
        attr(cls='body')
        p('Lorem ipsum..')

  return doc