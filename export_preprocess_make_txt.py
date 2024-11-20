from pia.ai.tasks.T2VRet.models.clip4clip.main import VisualModel 
from pia.ai.tasks.T2VRet.base import T2VRetConfig 
from pia.model import PiaTorchModel, PiaONNXTensorRTModel

from utils import normalized_transform, non_normalized_transform, LetterBox
from tile_custom_setting import ROI

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

visual_model = VisualModel(ret_model)

video = "falldown_01.mp4"
# video = "Abuse002_x264.mp4"

vid = cv2.VideoCapture(video)
TILE_SIZE = "M"

onnx_model = PiaONNXTensorRTModel("onnx_models/2024_10_11_2131_RealDiv255Include.onnx", "cuda")
# onnx_model = PiaONNXTensorRTModel("2024_10_11_0011_OnlyPreNorm_TorchNorm_visual.onnx", "cuda")
# onnx_model = PiaONNXTensorRTModel("2024_10_10_0011_OnlyPreNorm_WithCustomNorm_visual.onnx", "cuda")

# engine_model = PiaONNXTensorRTModel("div_255_pre_process_visual.engine", "cuda")
# txt_save_root = "logs/after_check_2024_10_08_abuse_vector_with_sub_div_layer"
# Keep dim 은 같ㅇ고 
# transform, custom transform은 같음 
# file_name = f"2024_10_10_2258_VideoFalls_{TILE_SIZE}_ExceptBothNorm"
file_name = f"2024_10_11_2319_VideoFalls_{TILE_SIZE}_test"
txt_save_root = f"logs/"
# txt_save_root = f"logs/2024_10_10_2136_video_prepost_result_TileSize_{TILE_SIZE}_NormCustom"
Path (txt_save_root).mkdir(exist_ok=True, parents=True)
origin_file = open(os.path.join(txt_save_root, f"{file_name}_python.txt"), 'w')
# onnx_file = open(os.path.join(txt_save_root, f"{file_name}_onnx.txt"), 'w') 
# normed_onnx_file = open(os.path.join(txt_save_root, f"{file_name}_normed_onnx.txt"), 'w') 
# engine_file = open("engine_origin.txt", "w")

resize_transform = LetterBox(new_shape=(224,224), auto=False, scaleFill=False, scaleup=False)

while True :
    ret, frame = vid.read()
    if ret : 
        with torch.no_grad():
            # tiling 코드 필요 
            origin_frame = frame.copy()
            origin_frame = resize_transform(image=origin_frame)
            # origin_frame = normalized_transform(origin_frame)
            origin_frame = non_normalized_transform(origin_frame)
            # origin_frame = torch.tensor(non_normalized_transform(origin_frame), dtype=torch.float32)[None, ::] # / 255 #.to("cuda") / 255

            print(frame.shape)
            # frame = frame.reshape(1,1,240, 320, 3)
            # tiles = []
            # for roi in ROI[TILE_SIZE]: 
            #     tile = frame[roi[0]:roi[1], roi[2]:roi[3]]
            #     tiles.append(tile)
            # frame = np.array(tiles, dtype=np.uint8)
            # frame = frame.reshape(1,1,1080, 1920, 3)
            # frame = tile_videos(videos_sequenced=frame, tile_size=TILE_SIZE)
            # batch, tile, sequence, height, width, channel = frame.shape
            # # # print(f"")
            # frame = frame.reshape(tile, height, width, channel)

            # frame = [resize_transform(image=f) for f in frame ] 
            # # frame = torch.stack([torch.tensor(non_normalized_transform(f), dtype=torch.float32) for f in frame])
            # frame = torch.stack([normalized_transform(f) for f in frame])
            # frame = torch.stack([f for f in frame])
            # input_dummy = non_normalized_transform(frame).to("cuda")[None, ::]

            # concat frames 
            # print(origin_frame)
            # input_dummy = torch.concat((origin_frame[None,::], frame))

            visual_output = visual_model(origin_frame.to("cuda"))
            # visual_output = visual_model(input_dummy.to('cuda'))
            visual_output = visual_output.type(torch.float16)
            # onnx_output = onnx_model(origin_frame)
            # onnx_output = onnx_model(input_dummy).type(torch.float16)
            # normed_onnx_output = onnx_output / onnx_output.norm(dim=-1, keepdim=True)
            # normed_visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
            # engine_output = engine_model(input_dummy)
        
        for v in visual_output:
            print(v.norm())
            origin_file.write(str(v.tolist())[1:-2] + "\n")
        # for v in onnx_output:
        #     onnx_file.write(str(v.tolist())[1:-2] + "\n")
        # for v in normed_onnx_output:
        #     normed_onnx_file.write(str(v.tolist())[1:-2] + "\n")

        # onnx_file.close()
        # origin_file.close()

        break
        # for v in normed_visual_output:
        #     normed_file.write(str(v.tolist())[1:-2] + "\n")
        
        # break 
        # engine_file.write(str(engine_output[0].tolist())[1:-2] + "\n")

    else : 
        break 

