from pia.ai.tasks.T2VRet.models.clip4clip.main import VisualModel 
from pia.ai.tasks.T2VRet.base import T2VRetConfig 
from pia.model import PiaTorchModel, PiaONNXTensorRTModel

from utils import normalized_transform, non_normalized_transform, LetterBox
from collections import deque
import numpy as np 
import os 
import cv2 
import torch 

from tile import tile_videos
from pathlib import Path 


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
video = "falldown_01.mp4"
vid = cv2.VideoCapture(video)
resize_transform = LetterBox(new_shape=(224,224), auto=False, scaleFill=False, scaleup=False)
maxlen = 12 
queue = deque(maxlen=maxlen)
visual_model = VisualModel(ret_model)

while True :
    ret, frame = vid.read()
    if ret : 
        with torch.no_grad():
            frame = resize_transform(image=frame)
            frame = normalized_transform(frame)
            # queue.append(frame)
            # if len(queue) < maxlen:
            #     continue
            visual_output = visual_model(frame)
            
            # sim = ret_model(
            #     video=torch.tensor(np.array(queue), dtype=torch.float32)[None,::].to("cuda"),
            #     text="The man is falldown")
            # print(sim)
