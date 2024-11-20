import onnx
from onnx import numpy_helper
from collections import defaultdict
import numpy as np 
from ast import literal_eval

file1 = "logs/2024_10_11_1748_VideoFalls_M_0820_Div255_onnx.txt"
file2 = "logs/2024_10_11_1748_VideoFalls_M_0820_Div255_python.txt"

data = defaultdict(dict)
for i, f in enumerate([file1, file2]):
    with open(f, "r") as rd : 
        # 문자열을 쉼표 기준으로 분리한 후, 각 요소를 float로 변환
        data_float = [float(i) for i in rd.readlines()[0].strip().split(', ')]

        # NumPy 배열로 변환
        data[i] = np.array(data_float)

print(1)
data 

