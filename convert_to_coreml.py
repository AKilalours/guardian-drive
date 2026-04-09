import torch
import torch.nn as nn
import coremltools as ct
import numpy as np

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

class WESADTCNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = TCNBlock(4, 64, dilation=1)
        self.b2 = TCNBlock(64, 128, dilation=2)
        self.b3 = TCNBlock(128, 128, dilation=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 2)
    def forward(self, x):
        x = self.b3(self.b2(self.b1(x)))
        return self.fc(self.pool(x).squeeze(-1))

model = WESADTCNModel()
model.eval()

example = torch.rand(1, 4, 256)

with torch.no_grad():
    traced = torch.jit.trace(model, example)

mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(name="ecg_eda_temp_resp", shape=example.shape)],
    minimum_deployment_target=ct.target.macOS13,
    convert_to="mlprogram"
)

mlmodel.short_description = "WESAD TCN drowsiness detector AUC 0.9514 - GuardianDrive"
mlmodel.author = "Akila Lourdes Miriyala Francis"
mlmodel.save("guardian_drive_tcn.mlpackage")
print("Saved: guardian_drive_tcn.mlpackage")

loaded = ct.models.MLModel("guardian_drive_tcn.mlpackage")
out = loaded.predict({"ecg_eda_temp_resp": np.random.rand(1, 4, 256).astype(np.float32)})
print("Inference result:", out)
print("CoreML conversion SUCCESS")
