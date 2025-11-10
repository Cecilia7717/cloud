import torch
from pathlib import Path
from utils.quant_utils.export.export_qonnx import export_model_qonnx

def save_quantized_model(model, save_path, input_shape):
    model.eval()

    # find model device
    device = next(model.parameters()).device

    # dummy input
    x = torch.randn(*input_shape).to(device)

    save_path = Path(save_path)

    export_model_qonnx(
        model=model,
        device=device,
        inp=x,
        export_path=save_path
    )

    model.train()
