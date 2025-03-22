#include "tensor.hpp"
#include "cudaOP.cuh"
#include <iostream>

// 特化 cast_to 方法实现
template <>
template <>
Tensor<half> Tensor<float>::cast_to<half>() const {
  // 创建空的 FP16 张量
  std::vector<half> empty_data(numel(), __float2half(0.0f));
  Tensor<half> result(std::move(empty_data), sizes(), device_);
  
  if (device_ == Device::CUDA) {
    // 使用 CUDA 进行转换
    for (size_t i = 0; i < numel(); ++i) {
      result.data_ptr()[i] = __float2half(data_ptr()[i]);
    }
  } else {
    // CPU 实现
    std::vector<half> cpu_data(numel());
    for (size_t i = 0; i < numel(); ++i) {
      cpu_data[i] = __float2half(data_->at(offset_ + i));
    }
    
    // 创建 CPU 张量并移动到正确的设备
    Tensor<half> cpu_result(std::move(cpu_data), sizes());
    if (device_ == Device::CUDA) {
      cpu_result.cuda();
    }
    result = cpu_result;
  }
  
  return result;
}

template <>
template <>
Tensor<float> Tensor<half>::cast_to<float>() const {
  // 创建空的 float 张量
  std::vector<float> empty_data(numel(), 0.0f);
  Tensor<float> result(std::move(empty_data), sizes(), device_);
  
  if (device_ == Device::CUDA) {
    // 使用 CUDA 进行转换
    for (size_t i = 0; i < numel(); ++i) {
      result.data_ptr()[i] = __half2float(data_ptr()[i]);
    }
  } else {
    // CPU 实现
    std::vector<float> cpu_data(numel());
    for (size_t i = 0; i < numel(); ++i) {
      cpu_data[i] = __half2float(data_->at(offset_ + i));
    }
    
    // 创建 CPU 张量并移动到正确的设备
    Tensor<float> cpu_result(std::move(cpu_data), sizes());
    if (device_ == Device::CUDA) {
      cpu_result.cuda();
    }
    result = cpu_result;
  }
  
  return result;
}

// 类似地实现 BF16 转换
template <>
template <>
Tensor<__nv_bfloat16> Tensor<float>::cast_to<__nv_bfloat16>() const {
  // 创建空的 BF16 张量
  std::vector<__nv_bfloat16> empty_data(numel(), __float2bfloat16(0.0f));
  Tensor<__nv_bfloat16> result(std::move(empty_data), sizes(), device_);
  
  if (device_ == Device::CUDA) {
    // 使用 CUDA 进行转换
    for (size_t i = 0; i < numel(); ++i) {
      result.data_ptr()[i] = __float2bfloat16(data_ptr()[i]);
    }
  } else {
    // CPU 实现
    std::vector<__nv_bfloat16> cpu_data(numel());
    for (size_t i = 0; i < numel(); ++i) {
      cpu_data[i] = __float2bfloat16(data_->at(offset_ + i));
    }
    
    // 创建 CPU 张量并移动到正确的设备
    Tensor<__nv_bfloat16> cpu_result(std::move(cpu_data), sizes());
    if (device_ == Device::CUDA) {
      cpu_result.cuda();
    }
    result = cpu_result;
  }
  
  return result;
}

template <>
template <>
Tensor<float> Tensor<__nv_bfloat16>::cast_to<float>() const {
  // 创建空的 float 张量
  std::vector<float> empty_data(numel(), 0.0f);
  Tensor<float> result(std::move(empty_data), sizes(), device_);
  
  if (device_ == Device::CUDA) {
    // 使用 CUDA 进行转换
    for (size_t i = 0; i < numel(); ++i) {
      result.data_ptr()[i] = __bfloat162float(data_ptr()[i]);
    }
  } else {
    // CPU 实现
    std::vector<float> cpu_data(numel());
    for (size_t i = 0; i < numel(); ++i) {
      cpu_data[i] = __bfloat162float(data_->at(offset_ + i));
    }
    
    // 创建 CPU 张量并移动到正确的设备
    Tensor<float> cpu_result(std::move(cpu_data), sizes());
    if (device_ == Device::CUDA) {
      cpu_result.cuda();
    }
    result = cpu_result;
  }
  
  return result;
} 