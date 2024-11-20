import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import os

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 엔진 파일 로드 함수
def load_engine(engine_file_path):
    if not os.path.exists(engine_file_path):
        print(f"Error: {engine_file_path} does not exist.")
        return None
    with open(engine_file_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            print("Error: Failed to load the engine.")
        return engine

# 입력 데이터 전처리 (필요에 따라 수정)
def preprocess_input(input_data):
    # 예시: 입력 데이터가 np.ndarray인 경우
    return np.ascontiguousarray(input_data, dtype=np.float32)

# 추론 수행 함수
def do_inference(engine, input_data, batch_size=1):
    if engine is None:
        print("Error: Engine is not loaded.")
        return None
    
    # Execution context 생성
    context = engine.create_execution_context()

    # 입출력 크기 가져오기
    input_shape = engine.get_binding_shape(0)
    output_shape = engine.get_binding_shape(1)
    print(input_shape, output_shape)

    # 메모리 할당
    d_input = cuda.mem_alloc(input_data.nbytes)
    output_data = np.empty(output_shape, dtype=np.float32)
    d_output = cuda.mem_alloc(output_data.nbytes)

    # 스트림 생성
    stream = cuda.Stream()

    # 입력 데이터를 GPU로 복사
    cuda.memcpy_htod_async(d_input, input_data, stream)

    # 추론 실행
    context.execute_async(batch_size=batch_size, bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)

    # 출력을 CPU로 복사
    cuda.memcpy_dtoh_async(output_data, d_output, stream)
    stream.synchronize()

    return output_data

# 예시 사용법
engine_file_path = "div_255_pre_process_visual.engine"
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)  # 예시 입력 데이터 (필요에 맞게 수정)

# 엔진 로드
engine = load_engine(engine_file_path)

# TensorRT에서 기대하는 입력 크기 확인
input_shape = engine.get_binding_shape(0)  # (batch_size, channels, height, width)
print("Expected input shape:", input_shape)

# 입력 데이터 크기 수정
input_data = np.random.rand(*input_shape).astype(np.float32)  # 엔진이 기대하는 입력 크기로 설정

# 추론 수행
output = do_inference(engine, input_data)
print("Inference output:", output)

if engine is not None:
    # 입력 전처리
    input_data = preprocess_input(input_data)

    # 추론 수행
    output = do_inference(engine, input_data)
    if output is not None:
        print("Inference output:", output)
else:
    print("Failed to load engine. Exiting.")
