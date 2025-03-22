cmake_minimum_required(VERSION 3.15)
project(llm_infer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# 获取 Python 版本
execute_process(
    COMMAND python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    OUTPUT_VARIABLE PYTHON_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(PYBIND11_PYTHON_VERSION ${PYTHON_VERSION})

# 设置 pybind11 的路径 (安装后使用上面命令获取的实际路径)
set(pybind11_DIR "/mnt/3T_disk2/chenqi/conda_envs/LearningInfiniTensor/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

# 设置 Torch 的路径 (使用上面命令获取的实际路径)
set(Torch_DIR "/mnt/3T_disk2/chenqi/conda_envs/LearningInfiniTensor/lib/python3.10/site-packages/torch/share/cmake/Torch")
find_package(Torch REQUIRED)

include_directories(backend/cpp/include)

pybind11_add_module(model_bridge 
    interface/model_bridge.cpp
    backend/cpp/src/llama_decode.cpp
    backend/cpp/src/llama_prefill.cpp
    backend/cpp/src/inference.cpp
    backend/cpp/src/thread_pool.cpp
    backend/cpp/src/cudaOP.cu
    backend/cpp/src/llama.cpp
    backend/cpp/src/tensor.cpp
)

target_link_libraries(model_bridge PRIVATE ${TORCH_LIBRARIES})
target_compile_options(model_bridge PRIVATE -mavx2 -mfma)

# 启用 CUDA 分离式编译
set_target_properties(model_bridge PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
