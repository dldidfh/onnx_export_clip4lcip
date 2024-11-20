from pia.ai.tasks.T2VRet.models.clip4clip.main import VisualModel 
from pia.ai.tasks.T2VRet.base import T2VRetConfig 
from pia.model import PiaTorchModel 

from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToPILImage,
    ToTensor,
)

import os 
import torch 
import numpy as np 
import cv2 


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

video = "Abuse002_x264.mp4"

vid = cv2.VideoCapture(video)
txt_save_path = "non_pre_normalize_visual.csv"
file_obj = open(txt_save_path, "w")

new_transform = Compose(
            [
                ToPILImage(),
                Resize(
                    224,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                CenterCrop(224),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

while True : 
    ret, frame = vid.read()
    if ret : 
        # video = frame[None,None,::]
        # B, F, H, W, C = video.shape
        # video = video.reshape(B * F, H, W, C)

        # # must do cvtColor - need channel convert
        # video_tensor = torch.stack(
        #     [new_transform(cv2.cvtColor(v, cv2.COLOR_BGR2RGB)) for v in video],
        #     dim=0,
        # )
        # video_tensor = video_tensor.reshape(
        #     (
        #         B,
        #         F,
        #         C,
        #         224,
        #         224,
        #     )
        # ).to("cuda")            
        # pre_processed_frame, mask =ret_model.video_preprocess(frame[None,::])
        # pre_processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # pre_processed_frame = new_transform(pre_processed_frame).to("cuda")
        # pre_processed_frame1, mask =ret_model.video_preprocess(frame[None,None,::])

        vis_output_1 = visual_model(frame[None,::])
        # vis_output_1 = visual_model(frame[None,::])
        vis_output_2 = ret_model(video=frame[None,::])

        normed_visual_output = vis_output_1 / vis_output_1.norm(dim=-1, keepdim=True)

        torch.equal(vis_output_2, normed_visual_output)
        # file_obj.write()
    else:break