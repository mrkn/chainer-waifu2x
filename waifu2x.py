import sys
import json
import math

import numpy as np
from PIL import Image

import chainer
import chainer.functions as F
from chainer import cuda
from chainer import optimizers

gpu_device_id = 0
try:
    cuda.check_cuda_available()
    use_gpu = True
    cuda.get_device(gpu_device_id).use()
except:
    use_gpu = False

waifu2x_model_file = 'data/anime_style_art_rgb_scale2.0x_model.json'
with open(waifu2x_model_file) as fp:
    model_params = json.load(fp)

def make_Convolution2D(params):
    func = F.Convolution2D(
        params['nInputPlane'],
        params['nOutputPlane'],
        (params['kW'], params['kH'])
    )
    func.b = np.float32(params['bias'])
    func.W = np.float32(params['weight'])
    return func

model = chainer.FunctionSet()
for i, layer_params in enumerate(model_params):
    function = make_Convolution2D(layer_params)
    setattr(model, "conv{}".format(i + 1), function)

if use_gpu:
    model.to_gpu()

steps = len(model_params)
pad_size = steps * 2

def forward(x):
    h = x
    for i in range(1, steps):
        key = 'conv{}'.format(i)
        h = F.leaky_relu(getattr(model, key)(h), 0.1)
    key = 'conv{}'.format(steps)
    y = getattr(model, key)(h)
    return y

def scale_image(image, block_offset, block_size=128):
    image = image.resize((2 * image.size[0], 2*image.size[1]), resample=Image.NEAREST)

    x_data = np.asarray(image).transpose(2, 0, 1).astype(np.float32)
    x_data /= 255

    output_size = block_size - block_offset * 2

    h_blocks = int(math.floor(x_data.shape[1] / output_size)) + (0 if x_data.shape[1] % output_size == 0 else 1)
    w_blocks = int(math.floor(x_data.shape[2] / output_size)) + (0 if x_data.shape[2] % output_size == 0 else 1)

    h = block_offset + h_blocks * output_size + block_offset
    w = block_offset + w_blocks * output_size + block_offset
    pad_h1 = block_offset
    pad_w1 = block_offset
    pad_h2 = (h - block_offset) - x_data.shape[1]
    pad_w2 = (w - block_offset) - x_data.shape[2]

    x_data = np.pad(x_data, ((0, 0), (pad_h1, pad_h2), (pad_w1, pad_w2)), 'edge')
    result_data = np.zeros_like(x_data)

    for i in range(0, x_data.shape[1], output_size):
        if i + block_size > x_data.shape[1]:
            continue
        for j in range(0, x_data.shape[2], output_size):
            if j + block_size > x_data.shape[2]:
                continue
            block = x_data[:, i:(i + block_size), j:(j + block_size)]
            block = np.reshape(block, (1,) + block.shape)
            if use_gpu:
                block = cuda.to_gpu(block)

            x = chainer.Variable(block)
            y = forward(x)
            y_data = cuda.to_cpu(y.data)[0]

            result_data[
                    :,
                    (i + block_offset):(i + block_offset + output_size),
                    (j + block_offset):(j + block_offset + output_size)
                ] = y_data

    result_data = result_data[
            :,
            (pad_h1 + 1):(result_data.shape[1] - pad_h2 - 1),
            (pad_w1 + 1):(result_data.shape[2] - pad_w2 - 1)]
    result_data[result_data < 0] = 0
    result_data[result_data > 1] = 1
    result_data *= 255

    result_image = Image.fromarray(np.uint8(result_data).transpose(1, 2, 0))
    return result_image

image = Image.open(sys.argv[1]).convert('RGB')
scaled_image = scale_image(image, steps)

scaled_image.save(sys.argv[2])
