from torch import FloatTensor, softmax
def get_similarity(
  mode0: FloatTensor,
  mode1: FloatTensor,
) -> FloatTensor:
  return softmax(mode0 @ mode1.T, dim=-1)