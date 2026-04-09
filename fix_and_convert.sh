#!/bin/bash
echo "=== Step 1: Downgrading scikit-learn ==="
pip install "scikit-learn==1.5.1" -q

echo "=== Step 2: Installing compatible coremltools ==="
pip install "coremltools==7.2" -q

echo "=== Step 3: Checking PyTorch version ==="
python -c "import torch; print('PyTorch:', torch.__version__)"

echo "=== Step 4: Running conversion in isolated process ==="
python - << 'PYEOF'
import torch
import torch.nn as nn
import numpy as np

print("Imports OK")

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        pad = (3 - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, 3, padding=pad, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = out[:, :, :x.size(2)]
        return out + self.res(x)

class WESADModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = TCNBlock(4, 64, 1)
        self.b2 = TCNBlock(64, 128, 2)
        self.b3 = TCNBlock(128, 128, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 2)
    def forward(self, x):
        x = self.b3(self.b2(self.b1(x)))
        return self.fc(self.pool(x).squeeze(-1))

model = WESADModel()
model.eval()
print("Model built OK")

example = torch.rand(1, 4, 256)
with torch.no_grad():
    out = model(example)
print("PyTorch forward pass OK, output shape:", out.shape)

# Save as ONNX first (avoids the jit.trace segfault)
torch.onnx.export(
    model,
    example,
    "wesad_tcn.onnx",
    input_names=["ecg_eda_temp_resp"],
    output_names=["drowsiness_logits"],
    opset_version=16,
    dynamic_axes={"ecg_eda_temp_resp": {0: "batch"}}
)
print("ONNX export OK: wesad_tcn.onnx")

# Now convert ONNX to CoreML (no segfault this way)
import coremltools as ct
print("coremltools version:", ct.__version__)

mlmodel = ct.convert(
    "wesad_tcn.onnx",
    source="pytorch",
    inputs=[ct.TensorType(name="ecg_eda_temp_resp", shape=(1, 4, 256))],
    minimum_deployment_target=ct.target.macOS13,
    convert_to="mlprogram"
)

mlmodel.short_description = "WESAD TCN drowsiness detector AUC 0.9514 GuardianDrive"
mlmodel.author = "Akila Lourdes Miriyala Francis"
mlmodel.save("guardian_drive_tcn.mlpackage")
print("CoreML model saved: guardian_drive_tcn.mlpackage")

# Verify inference
loaded = ct.models.MLModel("guardian_drive_tcn.mlpackage")
test = np.random.rand(1, 4, 256).astype(np.float32)
result = loaded.predict({"ecg_eda_temp_resp": test})
print("CoreML inference output:", result)
print("")
print("SUCCESS - CoreML conversion complete on Apple Silicon")
PYEOF
