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


class GanEncoderModel(nn.Module):
    """GanEncoder Model."""

    def __init__(self):
        """Init model."""

        super(GanEncoderModel, self).__init__()

    def forward(self, x):
        """Forward."""

        return x


def model_load(model, path):
    """Load model."""

    if not os.path.exists(path):
        print("Model '{}' does not exist.".format(path))
        return

    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    target_state_dict = model.state_dict()
    for n, p in state_dict.items():
        if n in target_state_dict.keys():
            target_state_dict[n].copy_(p)
        else:
            raise KeyError(n)


def model_save(model, path):
    """Save model."""

    torch.save(model.state_dict(), path)


def export_onnx():
    """Export onnx model."""

    import onnx
    import onnxruntime
    from onnx import optimizer
    import numpy as np

    onnx_file_name = "output/image_ganencoder.onnx"
    model_weight_file = 'models/ImageGanEncoder.pth'
    dummy_input = torch.randn(1, 3, 512, 512)

    # 1. Create and load model.
    model_setenv()
    torch_model = get_model(model_weight_file)
    torch_model.eval()

    # 2. Model export
    print("Export model ...")

    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {'input': {2: "height", 3: 'width'},
                    'output': {2: "height", 3: 'width'}
                    }

    torch.onnx.export(torch_model, dummy_input, onnx_file_name,
                  input_names=input_names,
                  output_names=output_names,
                  verbose=True,
                  opset_version=11,
                  keep_initializers_as_inputs=False,
                  export_params=True,
                  dynamic_axes=dynamic_axes)

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

    model_weight_file = 'models/ImageGanEncoder.pth'

    model_setenv()
    torch_model = get_model(model_weight_file)
    torch_model.eval()

    onnx_file_name = "output/image_ganencoder.onnx"
    onnxruntime_engine = onnxruntime.InferenceSession(onnx_file_name)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    dummy_input = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        torch_output = torch_model(dummy_input)
    onnxruntime_inputs = {onnxruntime_engine.get_inputs()[0].name: to_numpy(dummy_input)}
    onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)
    np.testing.assert_allclose(to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-02, atol=1e-02)
    print("Example1: Onnx model has been tested with ONNXRuntime, the result looks good !")

    # Test dynamic axes
    dummy_input = torch.randn(1, 3, 512, 511)
    with torch.no_grad():
        torch_output = torch_model(dummy_input)
    onnxruntime_inputs = {onnxruntime_engine.get_inputs()[0].name: to_numpy(dummy_input)}
    onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)
    np.testing.assert_allclose(to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-02, atol=1e-02)
    print("Example2: Onnx model has been tested with ONNXRuntime, the result looks good!")

    dummy_input = torch.randn(1, 3, 1024, 1024)
    with torch.no_grad():
        torch_output = torch_model(dummy_input)
    onnxruntime_inputs = {onnxruntime_engine.get_inputs()[0].name: to_numpy(dummy_input)}
    onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)
    np.testing.assert_allclose(to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-02, atol=1e-02)
    print("Example3: Onnx model has been tested with ONNXRuntime, the result looks good!")


def export_torch():
    """Export torch model."""

    script_file = "output/iamge_ganencoder.pt"
    weight_file = "models/ImageGanEncoder.pth"

    # 1. Load model
    print("Loading model ...")
    model = get_model(weight_file)
    model.eval()

    # 2. Model export
    print("Export model ...")
    dummy_input = torch.randn(1, 3, 512, 512)
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save(script_file)


def get_model(checkpoint):
    """Create encoder model."""

    model_setenv()
    model = GanEncoderModel()
    model_load(model, checkpoint)
    device = model_device()
    model.to(device)

    return model

def get_decoder_model(checkpoint):
    """Create decode model."""

    model_setenv()
    model = GanEncoderModel()
    model_load(model, checkpoint)
    device = model_device()
    model.to(device)

    return model    


class Counter(object):
    """Class Counter."""

    def __init__(self):
        """Init average."""

        self.reset()

    def reset(self):
        """Reset average."""

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update average."""

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(loader, model, optimizer, device, tag=''):
    """Trainning model ..."""

    total_loss = Counter()

    model.train()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            # Transform data to device
            images = images.to(device)
            targets = targets.to(device)

            predicts = model(images)

            # xxxx--modify here
            loss = nn.L1Loss(predicts, targets)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            # Update loss
            total_loss.update(loss_value, count)

            t.set_postfix(loss='{:.6f}'.format(total_loss.avg))
            t.update(count)

            # Optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return total_loss.avg


def valid_epoch(loader, model, device, tag=''):
    """Validating model  ..."""

    valid_loss = Counter()

    model.eval()

    with tqdm(total=len(loader.dataset)) as t:
        t.set_description(tag)

        for data in loader:
            images, targets = data
            count = len(images)

            # Transform data to device
            images = images.to(device)
            targets = targets.to(device)

            # Predict results without calculating gradients
            with torch.no_grad():
                predicts = model(images)

            # xxxx--modify here
            valid_loss.update(loss_value, count)
            t.set_postfix(loss='{:.6f}'.format(valid_loss.avg))
            t.update(count)


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

if __name__ == '__main__':
    """Test model ..."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--export', help="Export onnx model", action='store_true')
    parser.add_argument('--verify', help="Verify onnx model", action='store_true')

    args = parser.parse_args()

    if args.export:
        export_onnx()

    if args.verify:
        verify_onnx()
