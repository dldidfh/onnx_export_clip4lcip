
from pia.ai.tasks.T2VRet.models.clip4clip.main import VisualModel 
from utils import normalized_transform, non_normalized_transform, LetterBox
from pia.model import PiaTorchModel, PiaONNXTensorRTModel
from pia.ai.tasks.T2VRet.base import T2VRetConfig 

from tile import tile_videos
from pathlib import Path 
import torch 
import cv2 
import os 


clip4clip_path = os.getenv("CLIP4CLIP")
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
visual_model = VisualModel(ret_model)


video = "falldown_01.mp4"
vid = cv2.VideoCapture(video)
TILE_SIZE = "M"

file_name = f"2024_10_11_1824_model_output_test"
txt_save_root = f"logs/"
Path (txt_save_root).mkdir(exist_ok=True, parents=True)
origin_file = open(os.path.join(txt_save_root, f"{file_name}_python.txt"), 'w')
resize_transform = LetterBox(new_shape=(224,224), auto=False, scaleFill=False, scaleup=False)

while True :
    ret, frame = vid.read()
    if ret : 
        with torch.no_grad():
            origin_frame = frame.copy()
            origin_frame = resize_transform(image=origin_frame)
            # origin_frame = normalized_transform(origin_frame)
            # origin_frame = torch.tensor(normalized_transform(origin_frame), 
            #                             dtype=torch.float32)
            origin_frame = torch.tensor(non_normalized_transform(origin_frame), 
                                        dtype=torch.float32)

            print(frame.shape)
            visual_output = visual_model(origin_frame[None, ::].to("cuda"))
            visual_output = visual_output.type(torch.float16)
        for v in visual_output:
            print(v.norm())
            origin_file.write(str(v.tolist())[1:-2] + "\n")
        break 
    else: break 
origin_file.close()

