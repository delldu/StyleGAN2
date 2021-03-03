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
import os
import torch
import torch.nn as nn
from torchvision import models as models
import pdb

class GanEncoderModel(nn.Module):
    """GanEncoder Model."""

    def __init__(self, z_space_dim=512):
        """Init model."""

        super(GanEncoderModel, self).__init__()

        checkpoint = "models/ImageGanEncoder.pth"

        if os.path.exists(checkpoint):
            self.resnet50 = models.resnet50(pretrained=False)
        else:
            self.resnet50 = models.resnet50(pretrained=True)

        for param in self.resnet50.parameters():
            param.requires_grad = False

        fc_inputs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(nn.Linear(fc_inputs, z_space_dim), nn.Tanh())

        if os.path.exists(checkpoint):
            # load ...
            model_weights = torch.load(checkpoint)
            self.resnet50.load_state_dict(model_weights)
        else:
            # Saving weights
            torch.save(self.resnet50.state_dict(), checkpoint)

    def forward(self, x):
        """Forward."""
        return self.resnet50(x).unsqueeze(1)

def get_encoder():
    '''Get encoder'''

    model = GanEncoderModel()
    return model


def export_onnx():
    """Export onnx model."""

    import onnx
    import onnxruntime
    from onnx import optimizer
    import numpy as np

    onnx_file_name = "output/image_ganencoder.onnx"
    dummy_input = torch.randn(1, 3, 256, 256)

    # 1. Create and load model.
    torch_model = get_encoder()
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

    torch_model = get_encoder()
    torch_model.eval()

    onnx_file_name = "output/image_ganencoder.onnx"
    onnxruntime_engine = onnxruntime.InferenceSession(onnx_file_name)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        torch_output = torch_model(dummy_input)
    onnxruntime_inputs = {onnxruntime_engine.get_inputs()[0].name: to_numpy(dummy_input)}
    onnxruntime_outputs = onnxruntime_engine.run(None, onnxruntime_inputs)
    np.testing.assert_allclose(to_numpy(torch_output), onnxruntime_outputs[0], rtol=1e-02, atol=1e-02)
    print("Example: Onnx model has been tested with ONNXRuntime, the result looks good !")


def export_torch():
    """Export torch model."""

    script_file = "output/iamge_ganencoder.pt"

    # 1. Load model
    print("Loading model ...")
    model = get_encoder()
    model.eval()

    # 2. Model export
    print("Export model ...")
    dummy_input = torch.randn(1, 3, 256, 256)
    traced_script_module = torch.jit.trace(model, dummy_input)
    traced_script_module.save(script_file)

if __name__ == '__main__':
    """Onnx Tools ..."""
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--export', help="Export onnx model", action='store_true')
    parser.add_argument('--verify', help="Verify onnx model", action='store_true')
    parser.add_argument('--output', type=str, default="output", help="output folder")

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # export_torch()

    if args.export:
        export_onnx()

    if args.verify:
        verify_onnx()
