import numpy as np
import torch

def coord_asymmetric(x_resized, x_scale):
    return x_resized / x_scale

def coord_half_pixel(x_resized, x_scale, dummy):
    return ((x_resized + 0.5) / x_scale) - 0.5

def coord_pytorch_half_pixel(x_resized, x_scale, length_resized):
    return (x_resized + 0.5) / x_scale - 0.5 if length_resized > 1  else 0

orig = np.arange(16, dtype=float).reshape((4, 4))

a = torch.tensor([[orig]])
print(a.shape)
print(a)

ref = torch.nn.functional.interpolate(a, scale_factor=(2,2), mode="bilinear")

print(ref.shape)
print(ref)

def clamp(x, vmin, vmax):
    return max(vmin, min(x, vmax))

def bilinear_resize(a, scale_factors):
    def get_sample(x, y):
        x = clamp(x, 0, a.shape[1] - 1)
        y = clamp(y, 0, a.shape[0] - 1)
        # if x < 0 or x > a.shape[1]-1 or y < 0 or y > a.shape[0]-1:
        #     return 0
        return a[y, x]

    out = np.empty([s * scale_factors[i] for i, s in enumerate(a.shape)])
    H, W = out.shape
    for yo in range(H):
        for xo in range(W):
            # map the x,y coordinate back to original image, it may sit on some fractional position, aka, non-integer x,y
            # we are unable to directly use a[y, x] to get the sample value
            coord = coord_half_pixel(xo, scale_factors[1], W), coord_half_pixel(yo, scale_factors[0], H)

            # There is no sample exist. So we use some interpolate function to guess one from neighbour samples (aka, the pixels)
            # print(coord)
            x, y = coord
            x1 = int(x)
            y1 = int(y)
            x2 = int(x + 1)
            y2 = int(y + 1)
            p11 = get_sample(x1, y1)
            p12 = get_sample(x1, y2)
            p21 = get_sample(x2, y1)
            p22 = get_sample(x2, y2)

            # weights
            weight_x2 = x - x1
            weight_x1 = 1 - weight_x2
            weight_y2 = y - y1
            weight_y1 = 1 - weight_y2

            v = (weight_x1 * weight_y1) * p11 + (weight_x1 * weight_y2) * p12 + (weight_x2 * weight_y1) * p21 + (weight_x2 * weight_y2) * p22
            out[yo, xo] = v
    return out

my = bilinear_resize(orig, (2, 2))
print(my.shape)
print(my)

print(my - ref[0, 0].numpy())
