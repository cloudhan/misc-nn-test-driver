from typing import Tuple
import numpy as np

class Conv2D:
  """a general purpose 2d convolution (cross correlation kernel) forward for
    demonstration purpose.
    """

  def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel: Tuple[int, int],
      stride: Tuple[int, int],
      pad: Tuple[int, int],
      dilation: Tuple[int, int],
  ):
    self.C_in = in_channels
    self.C_out = out_channels
    self.K_w, self.K_h = kernel
    self.S_w, self.S_h = stride
    self.P_w, self.P_h = pad
    self.D_w, self.D_h = dilation

  def __call__(self, X, W, B):
    assert X.ndim == 4
    assert W.ndim == 4
    assert B.ndim == 1

    N, C_in, H_in, W_in = X.shape
    assert C_in == self.C_in
    C_out, C_in, K_h, K_w = W.shape
    assert C_out == self.C_out
    assert C_in == self.C_in
    assert K_h == self.K_h
    assert K_w == self.K_w

    H_out = Conv2D.out_shape(H_in, self.S_h, self.K_h, self.P_h, self.D_h)
    W_out = Conv2D.out_shape(W_in, self.S_w, self.K_w, self.P_w, self.D_w)

    Y = np.empty([N, self.C_out, H_out, W_out])
    for i in range(N):
      Conv2D._genernal_conv2d_kernel(
        X[i], H_in, W_in,
        W, C_out, C_in, K_h, K_w,
        Y[i], H_out, W_out,
        self.S_h, self.S_w,
        self.P_h, self.P_w,
        self.D_h, self.D_w
      ) # yapf:disable
    Y += B[:, np.newaxis, np.newaxis]

    return Y

  @staticmethod
  def out_shape(in_size, s, k, p, d):
    return (in_size + 2 * p - d * (k - 1) - 1) // s + 1

  @staticmethod
  def _genernal_conv2d_kernel(
      X, H_in, W_in, # input tensor
      W, C_out, C_in, K_h, K_w, # kernel weight
      Y, H_out, W_out, # output tensor
      S_h, S_w,
      P_h, P_w,
      D_h, D_w,
  ): # yapf:disable
    for co in range(C_out):  # for each output channel
      for h in range(H_out):  # for each output row
        for w in range(W_out):  # for each output element
          Y[co, h, w] = 0
          for ci in range(C_in):  # for each input channel
            for kh in range(K_h):  # for kernel width
              for kw in range(K_w):  # for kernel height
                hi = S_h*h + D_h*kh - P_h
                wi = S_w*w + D_w*kw - P_w
                if 0<= hi < H_in and 0<=wi< W_in:
                  weight = W[co, ci, kh, kw]
                  data = X[ci, hi, wi]
                  Y[co, h, w] += weight * data

if __name__ == "__main__":
  import torch
  import itertools

  kernels = (1,2,3)
  strides = (1,2,3)
  pads = (1,2,3)
  dilations = (1,2,3)

  for k,s,p,d in itertools.product(kernels, strides, pads, dilations):

    X = torch.rand([1, 2, 7, 7], dtype=torch.float32)
    W = torch.rand([5, 2, k, k], dtype=torch.float32)
    b = torch.linspace(1.2, 2.1, 5)
    Y_ref = torch.conv2d(X, W, b, stride=s, padding=p)

    my_conv2d = Conv2D(2, 5, kernel=(k,k), stride=(s,s), pad=(p,p), dilation=(1,1))
    Y_my = my_conv2d(X.numpy(), W.numpy(), b.numpy())
    # print(list(Y_ref.shape))
    # print(list(Y_my.shape))
    # print(np.abs(Y_my - Y_ref.numpy()) < 1e-6)
    np.testing.assert_allclose(Y_my, Y_ref.numpy(), rtol=1e-6, atol=1e-6)
