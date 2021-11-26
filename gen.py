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


class ConvAddZeros(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(3, 3, 3)
    self.conv2 = torch.nn.Conv2d(3, 3, 3)

  def forward(self, x):
    return self.conv1(x[0]) + self.conv2(x[1])

  def dummy_inputs(self):
    return [torch.rand([1, 3, 24, 24]), torch.rand([1, 3, 24, 24])]


class ConvAddConvAndAdd(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(3, 3, 3, padding=1)
    self.conv2 = torch.nn.Conv2d(3, 3, 3, padding=1)
    self.conv3 = torch.nn.Conv2d(3, 3, 3, padding=1)

  def forward(self, x):
    y = self.conv1(x[0]) + self.conv2(x[1])
    out1 = self.conv3(y)
    out2 = y + x[0]
    return [out1, out2]

  def dummy_inputs(self):
    return [torch.rand([1, 3, 24, 24]), torch.rand([1, 3, 24, 24])]


for name, instance in dict(locals()).items():
  if isinstance(instance, type) and issubclass(instance, torch.nn.Module):
    m = instance()
    torch.onnx.export(m, m.dummy_inputs(), name.lower() + ".onnx")
