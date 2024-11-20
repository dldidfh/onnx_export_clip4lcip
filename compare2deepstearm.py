import numpy as np 
import pandas as pd 

deepstream_output = "undong_log/withoutDiv255.csv"
# python_output = "logs/2024_10_07_whole_vectors/origin.csv"
python_output = "logs/2024_10_11_0114_VideoFalls_M_AmolRang_onnx.txt"

deepstream_output = pd.read_csv(deepstream_output, header=None)
python_output = pd.read_csv(python_output, header=None)

for i in range(len(deepstream_output)):
    # print(f"deepstream vector size : {np.linalg.norm(np.array(deepstream_output.loc[i]))}\t python vector size : {np.linalg.norm(np.array(python_output.loc[i]))}")
    deepstream_output_sep = np.array(deepstream_output.loc[i]) / np.linalg.norm(np.array(deepstream_output.loc[i]))
    # print(f"after norm deepstream vector size : {np.linalg.norm( deepstream_output_sep)}\t python vector size : {np.linalg.norm(np.array(python_output.loc[i]))}")
    python_output_norm = np.array(python_output.loc[i]) / np.linalg.norm(np.array(python_output.loc[i]))
    sim =  deepstream_output_sep @ python_output_norm
    print(f"sim score between tile {i} : {sim}")
print(1)

