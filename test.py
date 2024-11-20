import onnx
from onnx import numpy_helper
import numpy as np

# 1. ONNX 모델 로드
model = onnx.load('2024_10_11_0216_AMolRang.onnx')
onnx_file_save_path = "2024_10_11_0216_AMolRang_node_name.txt"
# model = onnx.load('old_models/div_255_pre_process_visual.onnx')
# onnx_file_save_path = "old_models/div_255_pre_process_visual_node_name.txt"
# 2. 가중치 추출
weights = {}
for tensor in model.graph.initializer:
    weights[tensor.name] = numpy_helper.to_array(tensor)

with open(onnx_file_save_path, "w") as wf :
   for node in model.graph.node:
        wf.write(f"{node.name} : {node.op_type}\n")
        print(f"노드 이름: {node.name}")
        print(f"연산자 유형: {node.op_type}")
        print(f"입력들: {node.input}")
        print(f"출력들: {node.output}\n")