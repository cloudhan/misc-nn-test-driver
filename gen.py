import types

import torch


def gen_binary_op(name, op, dummy_data):

  def forward(self, x):
    return {
        "+": lambda x: x[0] + x[1],
        "-": lambda x: x[0] - x[1],
        "*": lambda x: x[0] * x[1],
        "/": lambda x: x[0] / x[1],
    }[op](
        x)

  def dummy_inputs(self):
    return dummy_data

  return type(name, (torch.nn.Module,), dict(forward=forward, dummy_inputs=dummy_inputs))


APlusB = gen_binary_op("APlusB", "+", [torch.rand([64]), torch.rand([64])])
AMinusB = gen_binary_op("AMinusB", "-", [torch.rand([65537]), torch.rand([65537])])
AMulB = gen_binary_op("AMulB", "*", [torch.rand([65535]), torch.rand([65535])])
ADivB = gen_binary_op("ADivB", "/", [torch.rand([64023]), torch.rand([64023])])


class ConvAddZeros(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(3, 3, 3, padding=1)

  def forward(self, x):
    return self.conv1(x[0]) + x[1]

  def dummy_inputs(self):
    return [torch.rand([1, 3, 24, 24]), torch.zeros([1, 3, 24, 24])]


class ConvAddDev1(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(4, 1, 4, padding=0)
    self.conv1.weight = torch.nn.parameter.Parameter(torch.ones_like(self.conv1.weight))

  def forward(self, x):
    return self.conv1(x[0]) + x[1]

  def dummy_inputs(self):
    # return [torch.rand([1, 1, 4, 4]), torch.rand([1, 1, 4, 4])]
    return [torch.arange(64, dtype=torch.float32).reshape([1, 4, 4, 4]), torch.zeros([1, 1, 1, 1])]


class ConvAdd(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(3, 5, 3)

  def forward(self, x):
    return self.conv1(x[0]) + x[1]

  def dummy_inputs(self):
    return [torch.rand([1, 3, 15, 15]), torch.rand([1, 5, 13, 13])]


class ConvConvAdd(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(3, 3, 3)
    self.conv2 = torch.nn.Conv2d(3, 3, 3)

  def forward(self, x):
    return self.conv1(x[0]) + self.conv2(x[1])

  def dummy_inputs(self):
    return [torch.rand([1, 3, 24, 24]), torch.rand([1, 3, 24, 24])]


class DepthwiseConv(torch.nn.Module):

  def __init__(self):
    super().__init__()
    ci_per_group = 1
    co_per_group = 1
    self.group = 5
    self.dw_conv = torch.nn.Conv2d(ci_per_group * self.group, co_per_group * self.group, 3, groups=self.group)

  def forward(self, x):
    return self.dw_conv(x[0])

  def dummy_inputs(self):
    return [torch.rand([1, self.group, 5, 5])]


for name, instance in dict(locals()).items():
  if isinstance(instance, type) and issubclass(instance, torch.nn.Module):
    m = instance()
    torch.onnx.export(m, m.dummy_inputs(), name.lower() + ".onnx")
