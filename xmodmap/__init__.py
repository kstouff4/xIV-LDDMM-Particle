import torch
import numpy as np

# Set data type in: fromScratHamiltonianAT, analyzeOutput, getInput, initialize
use_cuda = torch.cuda.is_available()
if use_cuda:
    dtype = torch.cuda.FloatTensor  # DoubleTensor
    map_location = lambda storage, loc: storage.cuda(0)
else:
    dtype = torch.FloatTensor
    map_location = torch.device("cpu")

torch.set_default_tensor_type(dtype)


def normalize(x, p=1.0, dim=1):
    """
    Normalize a tensor or numpy array along a given dimension. Return a copy of the input.

    example:
    >>> a = np.row_stack((np.zeros((3,2)), np.ones((1, 2))))
    >>> normalize(a, p=2, dim=1)
    >>> b = torch.tensor(a)
    >>> normalize(b, p=2, dim=1)

    """
    if isinstance(x, torch.Tensor):
        return torch.nn.functional.normalize(x, p=p, dim=dim)
    else:
        tmp = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)
        gd = tmp.squeeze() > 0
        res = np.copy(x)
        res[gd] = res[gd] / tmp[gd]
        return res


#
# a = np.row_stack((np.zeros((3,2)), np.ones((1, 2))))
# print(a)
# print(a.__array_interface__['data'])
#
# b = normalize(a)
# print(b)
# print(b.__array_interface__['data'][0])
# print(a)
#
#
# c = torch.tensor(a)
# print(c.data_ptr())
#
# d = normalize(c)
# print(d.data_ptr())
#
# print(c)
# print(d)


import xmodmap.distance as distance
import xmodmap.deformation as deformation
import xmodmap.model as model