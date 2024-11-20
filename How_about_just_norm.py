import torch 
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    ToPILImage
)
import onnx
import onnxruntime as ort


class TestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transform = Compose(
            [
                ToPILImage(),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    def forward(self,x):
        return self.transform(x)
    
model = TestModel()
input_dummy = torch.rand(size=(1, 3,224,224), dtype=torch.float32)
onnx_save_path = "onnx_models/2024_10_11_2252_testmodel.onnx"

torch.onnx.export(
    model,
    input_dummy,
    # (torch.tensor(input_dummy, dtype=torch.uint8).to("cuda")),
    onnx_save_path,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input' : {0: 'batch_size',
                            },  
                'output': {0: 'batch_size',
                            }
                },
    opset_version=17
)

# ONNX 모델 로드 및 시각화
onnx_model = onnx.load(onnx_save_path)
onnx.checker.check_model(onnx_model)

# ONNX 모델 실행
ort_session = ort.InferenceSession(onnx_save_path)

ort_inputs = {
    ort_session.get_inputs()[0].name: input_dummy.numpy(),
              }

ort_outs = ort_session.run(None, ort_inputs)

python_output = model(input_dummy)

print(1)