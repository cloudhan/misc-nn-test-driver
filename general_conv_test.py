import pytest
import torch
import numpy as np

import general_conv as my

kernels = (1,2,3,5)
strides = (1,2,3)
pads = (1,2,3)
dilations = (1,2,3)

@pytest.mark.parametrize("k", kernels)
@pytest.mark.parametrize("s", strides)
@pytest.mark.parametrize("p", pads)
@pytest.mark.parametrize("d", dilations)
def test_conv(k, s, p, d):
  X = torch.rand([1, 2, 20, 20], dtype=torch.float32)
  W = torch.rand([5, 2, k, k], dtype=torch.float32)
  b = torch.linspace(1.2, 2.1, 5)
  Y_ref = torch.conv2d(X, W, b, stride=s, padding=p, dilation=d)

  my_conv2d = my.Conv2D(2, 5, kernel=(k,k), stride=(s,s), pad=(p,p), dilation=(d,d))
  Y_my = my_conv2d(X.numpy(), W.numpy(), b.numpy())
  # print(list(Y_ref.shape))
  # print(list(Y_my.shape))
  # print(np.abs(Y_my - Y_ref.numpy()) < 1e-6)
  np.testing.assert_allclose(Y_my, Y_ref.numpy(), rtol=1e-6, atol=1e-6)

if __name__ == "__main__":
  pytest.main()
