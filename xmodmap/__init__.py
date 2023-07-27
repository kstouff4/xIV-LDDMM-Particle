import torch


import numpy as np


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
