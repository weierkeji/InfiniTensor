#pragma once
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "inference.hpp"
#include "tensor.hpp"
#include "thread_pool.hpp"
class LlamaModel {
 public:
  LlamaModel(const std::unordered_map<std::string, Tensor<float>>& params,
             const std::unordered_map<std::string, int>& config,
             PrecisionType compute_precision = PrecisionType::FP32);
  bool verify_params() const;
  void print_model_info() const;

  // 前向计算
  Tensor<float> forward(const Tensor<uint32_t>* input, ThreadPool& thread_pool,
                        KVCache* kv_cache);
  Tensor<float> prefill(const Tensor<uint32_t>* input, ThreadPool& thread_pool,
                        KVCache* kv_cache);
  Tensor<float> prefill_cpu(const Tensor<uint32_t>* input, KVCache* kv_cache,
                            ThreadPool& thread_pool);
  Tensor<float> prefill_cuda(const Tensor<uint32_t>* input, KVCache* kv_cache);
  Tensor<float> forward_cpu(const Tensor<uint32_t>* input,
                            ThreadPool& thread_pool, KVCache* kv_cache);
  Tensor<float> forward_cuda(const Tensor<uint32_t>* input, KVCache* kv_cachel);

  std::vector<uint32_t> generate(const std::vector<uint32_t>& input_ids,
                                 size_t max_length, float temperature = 1.0f,
                                 float top_p = 0.9f, size_t top_k = 50);

  // Getter方法
  size_t get_n_layers() const { return n_layers_; }
  size_t get_max_seq_len() const { return max_seq_len_; }
  size_t get_head_dim() const { return dqkv_; }
  size_t get_n_kv_heads() const { return n_kv_h_; }
  uint32_t get_eos_token_id() const { return eos_token_id_; }

  // CUDA support methods
  LlamaModel& cuda();
  LlamaModel& cpu();
  Device device() const { return device_; }

  // 添加精度控制方法
  void set_compute_precision(PrecisionType precision);
  PrecisionType get_compute_precision() const { return compute_precision_; }

 private:
  // 基础参数
  size_t vocab_size_;
  size_t n_layers_;
  size_t n_q_h_;
  size_t n_kv_h_;
  size_t d_;     // hidden_size
  size_t dqkv_;  // hidden_size / num_attention_heads
  size_t di_;    // intermediate_size
  float eps_;
  float rope_theta_;
  size_t max_seq_len_;
  uint32_t bos_token_id_;
  uint32_t eos_token_id_;

  // 模型参数
  std::unordered_map<std::string, Tensor<float>> params_;
  Device device_;

  // 添加精度类型成员
  PrecisionType compute_precision_;
  
  // 添加低精度权重缓存
  std::unordered_map<std::string, Tensor<half>> fp16_params_;
  std::unordered_map<std::string, Tensor<__nv_bfloat16>> bf16_params_;
  
  // 转换权重到目标精度
  void convert_weights_to_precision(PrecisionType target_precision);
};