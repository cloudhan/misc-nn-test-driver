import types

import torch
import warnings

module_level_warning_catcher = warnings.catch_warnings()
warnings.simplefilter("ignore")

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


class ClipDev1(torch.nn.Module):

  def forward(self, x):
    return torch.clamp(x[0], min=10, max=11)

  def dummy_inputs(self):
    return [torch.zeros([29, 199, 14, 14])]

class ClipDev2(torch.nn.Module):

  def forward(self, x):
    return torch.clamp(x[0], min=0.0, max=1.0)

  def dummy_inputs(self):
    return [torch.zeros([1, 1, 256, 256])]

class ReluDev1(torch.nn.Module):

  def forward(self, x):
    return torch.relu(x[0])

  def dummy_inputs(self):
    return [torch.zeros([29, 199, 14, 14])]

class ConvDev1(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(192, 64, 1, padding=0)
    self.conv1.weight = torch.nn.parameter.Parameter(torch.ones_like(self.conv1.weight))
    self.conv1.bias = torch.nn.parameter.Parameter(torch.zeros_like(self.conv1.bias))

  def forward(self, x):
    return self.conv1(x[0])

  def dummy_inputs(self):
    return [torch.ones([1, 192, 14, 14])]


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


class DepthwiseConvDev1(torch.nn.Module):

  def __init__(self):
    super().__init__()
    ci_per_group = 1
    co_per_group = 1
    self.group = 7
    self.dw_conv = torch.nn.Conv2d(ci_per_group * self.group, co_per_group * self.group, 5, groups=self.group)
    self.dw_conv.weight = torch.nn.parameter.Parameter(
        torch.arange(self.dw_conv.weight.numel(), dtype=torch.float32).reshape(self.dw_conv.weight.shape))
    self.dw_conv.bias = torch.nn.parameter.Parameter(torch.zeros_like(self.dw_conv.bias))

  def forward(self, x):
    return self.dw_conv(x[0])

  def dummy_inputs(self):
    return [torch.ones([1, self.group, 5, 5])]


class DepthwiseConv(torch.nn.Module):

  def __init__(self):
    super().__init__()
    ci_per_group = 1
    co_per_group = 1
    self.group = 6
    self.dw_conv = torch.nn.Conv2d(ci_per_group * self.group, co_per_group * self.group, 3, groups=self.group)

  def forward(self, x):
    return self.dw_conv(x[0])

  def dummy_inputs(self):
    return [torch.rand([1, self.group, 5, 5])]

class MaxPoolDev1(torch.nn.Module):

  def forward(self, x):
    return torch.max_pool2d(x[0], (2,2), (2,2), (0,0), (1,1), False)

  def dummy_inputs(self):
    return [torch.zeros([29, 199, 14, 14])]


for name, instance in dict(locals()).items():
  if isinstance(instance, type) and issubclass(instance, torch.nn.Module):
    m = instance()
    torch.onnx.export(m, m.dummy_inputs(), name.lower() + ".onnx")
