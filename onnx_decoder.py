"""Create model."""# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 03月 02日 星期二 12:48:05 CST
# ***
# ************************************************************************************/
#

import math
import os
import sys

import torch
import torch.nn as nn
from tqdm import tqdm

from model import Generator


def model_load(model, path):
    """Load model."""

    if not os.path.exists(path):
        print("Model '{}' does not exist.".format(path))
        return

    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    state_dict = state_dict["g_ema"]

    target_state_dict = model.state_dict()
    for n, p in state_dict.items():
        if n in target_state_dict.keys():
            target_state_dict[n].copy_(p)
        else:
            raise KeyError(n)


def export_onnx():
    """Export onnx model."""

    import onnx
    import onnxruntime
    from onnx import optimizer
    import numpy as np

    onnx_file_name = "output/image_gandecoder.onnx"
    model_weight_file = 'models/ImageGanDecoder.pth'
    dummy_input = torch.randn(1, 512)

    # 1. Create and load model.
    model_setenv()
    torch_model = get_model(model_weight_file)
    torch_model.eval()

    # 2. Model export
    print("Export model ...")

    input_names = ["input"]
    output_names = ["output"]

    torch.onnx.export(torch_model, dummy_input, onnx_file_name,
                  input_names=input_names,
                  output_names=output_names,
                  verbose=True,
                  opset_version=11,
                  keep_initializers_as_inputs=False,
                  export_params=True)

    # 3. Optimize model
    print('Checking model ...')
    onnx_model = onnx.load(onnx_file_name)
    onnx.checker.check_model(onnx_model)
    # https://github.com/onnx/optimizer

    # 4. Visual model
    # python -c "import netron; netron.start('output/image_zoom.onnx')"

def verify_onnx():
    """Verify onnx model."""

    import onnxruntime
    import numpy as np

    model_weight_file = 'models/ImageGanDecoder.pth'

    model_setenv()
    torch_model = get_model(model_weight_file)
    torch_model.eval()

    onnx_file_name = "output/image_gandecoder.onnx"
    onnxruntime_engine = onnxruntime.InferenceSession(onnx_file_name)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    dummy_input = torch.randn(1, 512)
    with torch.no_grad():
        torch_output = torch_model(dummy_input)
    onnxruntime_inputs = {onnxruntime_engine.get_inputs()[0].name: to_numpy(dummy_input)}
    onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)
    np.testing.assert_allclose(to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-02, atol=1e-02)
    print("Example: Onnx model has been tested with ONNXRuntime, the result looks good !")


def get_model(checkpoint):
    """Create encoder model."""

    model_setenv()
    model = Generator(1024, 512, 8)
    model_load(model, checkpoint)
    device = model_device()
    model.to(device)
    return model


def model_device():
    """Please call after model_setenv. """

    return torch.device(os.environ["DEVICE"])


def model_setenv():
    """Setup environ  ..."""

    # random init ...
    import random
    random.seed(42)
    torch.manual_seed(42)

    # Set default device to avoid exceptions
    if os.environ.get("DEVICE") != "cuda" and os.environ.get("DEVICE") != "cpu":
        os.environ["DEVICE"] = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.environ["DEVICE"] == 'cuda':
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    print("Running Environment:")
    print("----------------------------------------------")
    print("  PWD: ", os.environ["PWD"])
    print("  DEVICE: ", os.environ["DEVICE"])

def export_torch():
    """Export torch model."""

    script_file = "output/image_gandecoder.pt"
    weight_file = "models/ImageGanDecoder.pth"

    # 1. Load model
    print("Loading model ...")
    model = get_model(weight_file)
    model.eval()

    # 2. Model export
    print("Export model ...")
    dummy_input = torch.randn(1, 512)
    traced_script_module = torch.jit.trace(model, dummy_input, _force_outplace=True)
    traced_script_module.save(script_file)


if __name__ == '__main__':
    """Test model ..."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--export', help="Export onnx model", action='store_true')
    parser.add_argument('--verify', help="Verify onnx model", action='store_true')

    args = parser.parse_args()

    # export_torch()

    if args.export:
        export_onnx()

    if args.verify:
        verify_onnx()
