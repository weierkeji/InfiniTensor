#include "llama.hpp"
#include "cudaOP.cuh"
#include <iostream>

void LlamaModel::convert_weights_to_precision(PrecisionType target_precision) {
  if (target_precision == PrecisionType::FP16) {
    // 转换权重到 FP16
    for (const auto& pair : params_) {
      const std::string& name = pair.first;
      const Tensor<float>& tensor = pair.second;
      
      // 直接使用 cast_to 方法
      fp16_params_[name] = tensor.cast_to<half>();
    }
  } else if (target_precision == PrecisionType::BF16) {
    // 转换权重到 BF16
    for (const auto& pair : params_) {
      const std::string& name = pair.first;
      const Tensor<float>& tensor = pair.second;
      
      // 直接使用 cast_to 方法
      bf16_params_[name] = tensor.cast_to<__nv_bfloat16>();
    }
  }
}

void LlamaModel::set_compute_precision(PrecisionType precision) {
  if (compute_precision_ != precision) {
    compute_precision_ = precision;
    
    // 如果切换到低精度，需要转换权重
    if (precision != PrecisionType::FP32) {
      convert_weights_to_precision(precision);
    }
  }
} 