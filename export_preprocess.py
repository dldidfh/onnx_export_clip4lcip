from models import VisualModel 
from pia.ai.tasks.T2VRet.base import T2VRetConfig 
from pia.model import PiaTorchModel 

from utils import normalized_transform, non_normalized_transform, LetterBox

import os 
import torch 
import numpy as np 
import onnx
import onnxruntime as ort
import cv2 
from pathlib import Path 

clip4clip_path = os.getenv("CLIP4CLIP")
# clip4clip_path = "/home/piawsa6000/work/model_zoo/models/clip4clip_ours/KTT/20241015_fine_tune_PIA_KTT.pt"

ret_config = T2VRetConfig(
    model_path=clip4clip_path,
    device="cuda",
    half=False,
    img_size=[224,224],
    max_words=32,
    temporal_size=12,
    frame_skip=15,
    clip_config="openai/clip-vit-base-patch32"
)

ret_model = PiaTorchModel(
    target_task="RET",
    target_model=0,
    config=ret_config
)
# resize_transform = LetterBox(new_shape=(224,224), auto=False, scaleFill=False, scaleup=False)
visual_model = VisualModel(ret_model)
visual_model = visual_model.float()

video = "video/falldown_01.mp4"
vid = cv2.VideoCapture(video)
ret, frame = vid.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
TILE_SIZE = "M"

save_file_name = f"2024_10_15_test3"
txt_save_root = f"logs/"
onnx_save_root = f"onnx_models"
Path (txt_save_root).mkdir(exist_ok=True, parents=True)
Path (onnx_save_root).mkdir(exist_ok=True, parents=True)


# frame = resize_transform(image=frame)
frame = cv2.resize(frame, dsize=(224,224), interpolation=cv2.INTER_LINEAR)

# frame = normalized_transform(frame)[None,::].to("cuda")
frame = non_normalized_transform(frame)[None,::].to("cuda")

with torch.no_grad() :
    vis_output = visual_model(frame)
vis_output = vis_output.cpu().detach().numpy()
onnx_save_path = f"{onnx_save_root}/{save_file_name}.onnx"

torch.onnx.export(
    visual_model,
    frame,
    onnx_save_path,
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input' : {0: 'batch_size',}, 'output': {0: 'batch_size',}},
    opset_version=17
) 

# ONNX 모델 로드 및 시각화
onnx_model = onnx.load(onnx_save_path)
onnx.checker.check_model(onnx_model)

# ONNX 모델 실행
ort_session = ort.InferenceSession(onnx_save_path)
ort_inputs = {ort_session.get_inputs()[0].name: frame.cpu().numpy()}
ort_outs = ort_session.run(None, ort_inputs)

check = np.allclose(ort_outs, vis_output, atol=1e-05, rtol=1e-04)

if not check:
    raise "Two vector is different"



origin_file = open(os.path.join(txt_save_root, f"{save_file_name}_python.txt"), 'w')
onnx_file = open(os.path.join(txt_save_root, f"{save_file_name}_onnx.txt"), 'w') 
for v in vis_output:
    origin_file.write(str(v.tolist())[1:-2] + "\n")
for v in ort_outs:
    onnx_file.write(str(v.tolist())[1:-2] + "\n")