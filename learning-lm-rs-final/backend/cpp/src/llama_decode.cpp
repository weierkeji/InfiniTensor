#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <string>

#include "avx_operators.hpp"
#include "cudaOP.cuh"
#include "inference.hpp"
#include "llama.hpp"
#include "operators.hpp"

// Debug print function for tensors
template <typename T>
void debugPrintTensor(const Tensor<T>& tensor, const std::string& tensor_name,
                      size_t num_to_print = 10) {
  std::cout << "[Debug] " << tensor_name << ":\n";

  // 1) Print shape
  std::cout << "  shape: [";
  for (auto s : tensor.sizes()) {
    std::cout << s << " ";
  }
  std::cout << "]\n";

  // 2) Print strides
  std::cout << "  strides: [";
  for (auto st : tensor.strides()) {
    std::cout << st << " ";
  }
  std::cout << "]\n";

  // 3) Print device
  std::cout << "  device: ";
  if (tensor.device() == Device::CPU) {
    std::cout << "CPU";
  } else if (tensor.device() == Device::CUDA) {
    std::cout << "CUDA";
  } else {
    std::cout << "UNKNOWN";
  }
  std::cout << "\n";

  // 4) Print elements starting from offset 312
  size_t offset = 21330;  // Custom offset for debugging
  size_t total_elements = tensor.numel();
  size_t n_print = (total_elements > offset)
                       ? std::min(num_to_print, total_elements - offset)
                       : 0;

  std::cout << "  elements from offset " << offset << " (" << n_print
            << " element(s)): ";
  if (tensor.device() == Device::CPU) {
    const T* ptr = tensor.data_ptr();
    for (size_t i = 0; i < n_print; i++) {
      std::cout << ptr[offset + i] << " ";
    }
    std::cout << "\n";
  } else {
    // Copy from GPU to CPU, then print
    std::vector<T> host_buffer(n_print);
    cudaError_t err = cudaMemcpy(host_buffer.data(), tensor.data_ptr() + offset,
                                 n_print * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      std::cout << "  [Error] cudaMemcpy failed\n";
      return;
    }
    for (size_t i = 0; i < n_print; i++) {
      std::cout << host_buffer[i] << " ";
    }
    std::cout << "\n";
  }
}

LlamaModel::LlamaModel(
    const std::unordered_map<std::string, Tensor<float>>& params,
    const std::unordered_map<std::string, int>& config,
    PrecisionType compute_precision)
    : params_(params), compute_precision_(compute_precision) {
  vocab_size_ = config.at("vocab_size");
  n_layers_ = config.at("num_hidden_layers");
  n_q_h_ = config.at("num_attention_heads");
  n_kv_h_ = config.at("num_key_value_heads");
  d_ = config.at("hidden_size");
  dqkv_ = d_ / n_q_h_;  // Using n_q_h_ to calculate dqkv_
  di_ = config.at("intermediate_size");
  eps_ = config.at("rms_norm_eps");
  rope_theta_ = config.at("rope_theta");
  max_seq_len_ = config.at("max_position_embeddings");
  bos_token_id_ = config.at("bos_token_id");
  eos_token_id_ = config.at("eos_token_id");

  // 如果使用低精度，初始化转换权重
  if (compute_precision_ != PrecisionType::FP32) {
    convert_weights_to_precision(compute_precision_);
  }
}

LlamaModel& LlamaModel::cuda() {
  if (device_ == Device::CUDA) {
    std::cout << "[LlamaModel::cuda] Model already on CUDA device" << std::endl;
    return *this;  // Already on CUDA
  }

  std::cout << "[LlamaModel::cuda] Moving model parameters to CUDA..."
            << std::endl;
  for (auto& pair : params_) {
    std::cout << "[LlamaModel::cuda] Moving " << pair.first << " to CUDA..."
              << std::endl;
    pair.second.cuda();

    // Verify if the parameter has been successfully moved to CUDA
    if (pair.second.device() != Device::CUDA) {
      std::string err = "Failed to move parameter " + pair.first + " to CUDA";
      std::cerr << "[LlamaModel::cuda] ERROR: " << err << std::endl;
      throw std::runtime_error(err);
    }
  }

  device_ = Device::CUDA;
  std::cout << "[LlamaModel::cuda] All parameters moved to CUDA successfully"
            << std::endl;
  return *this;
}

LlamaModel& LlamaModel::cpu() {
  if (device_ == Device::CPU) {
    return *this;  // Already on CPU
  }
  for (auto& pair : params_) {
    pair.second.cpu();
  }
  device_ = Device::CPU;
  return *this;
}

bool LlamaModel::verify_params() const {
  try {
    // Validate weights existence
    const std::vector<std::string> required_weights = {"embedding_table",
                                                       "lm_head", "rms_out_w"};
    for (const auto& weight : required_weights) {
      if (params_.find(weight) == params_.end()) {
        std::cerr << "Missing required weight: " << weight << std::endl;
        return false;
      }
    }

    // Validate weights and dimensions for each layer
    for (size_t i = 0; i < n_layers_; i++) {
      const std::vector<std::string> layer_weights = {
          "rms_att_w" + std::to_string(i), "wq" + std::to_string(i),
          "wk" + std::to_string(i),        "wv" + std::to_string(i),
          "wo" + std::to_string(i),        "rms_ffn_w" + std::to_string(i),
          "w_up" + std::to_string(i),      "w_down" + std::to_string(i),
          "w_gate" + std::to_string(i)};
      for (const auto& weight : layer_weights) {
        if (params_.find(weight) == params_.end()) {
          std::cerr << "Missing layer weight: " << weight << std::endl;
          return false;
        }
      }

      // Verify weight matrix dimensions
      const auto& wq = params_.at("wq" + std::to_string(i));
      const auto& wk = params_.at("wk" + std::to_string(i));
      const auto& wv = params_.at("wv" + std::to_string(i));
      const auto& wo = params_.at("wo" + std::to_string(i));

      if (wq.sizes()[1] != n_q_h_ * dqkv_ || wq.sizes()[0] != d_) {
        std::cerr << "Q weight matrix dimension mismatch at layer " << i
                  << std::endl;
        return false;
      }
      if (wk.sizes()[1] != n_kv_h_ * dqkv_ || wk.sizes()[0] != d_) {
        std::cerr << "K weight matrix dimension mismatch at layer " << i
                  << std::endl;
        return false;
      }
      if (wv.sizes()[1] != n_kv_h_ * dqkv_ || wv.sizes()[0] != d_) {
        std::cerr << "V weight matrix dimension mismatch at layer " << i
                  << std::endl;
        return false;
      }
      if (wo.sizes()[1] != d_ || wo.sizes()[0] != n_q_h_ * dqkv_) {
        std::cerr << "Output projection matrix dimension mismatch at layer "
                  << i << std::endl;
        return false;
      }
    }
    return true;
  } catch (const std::out_of_range& e) {
    std::cerr << "Missing weight: " << e.what() << std::endl;
    return false;
  }
}

void LlamaModel::print_model_info() const {
  std::cout << "Model Info:\n"
            << "vocab_size: " << vocab_size_ << "\n"
            << "n_layers: " << n_layers_ << "\n"
            << "n_q_h: " << n_q_h_ << "\n"
            << "n_kv_h: " << n_kv_h_ << "\n"
            << "hidden_size: " << d_ << "\n"
            << "head_dim: " << dqkv_ << "\n"
            << "intermediate_size: " << di_ << std::endl;
}

Tensor<float> LlamaModel::forward_cpu(const Tensor<uint32_t>* input,
                                      ThreadPool& thread_pool,
                                      KVCache* kv_cache) {
  const size_t seq_len = input->numel();
  const size_t n_groups = n_q_h_ / n_kv_h_;
  const size_t current_pos = kv_cache ? kv_cache->size() - seq_len : 0;

  std::vector<float> residual_data(seq_len * d_);
  std::vector<float> hidden_states_data(seq_len * d_);
  Tensor<float> residual(std::move(residual_data), {seq_len, d_});
  Tensor<float> hidden_states(std::move(hidden_states_data), {seq_len, d_});

  // Print input tensor details
  // debugPrintTensor(*input, "Input tensor");

  OP::gather(&residual, input, &params_.at("embedding_table"));

  // Debug print residual after embedding
  // debugPrintTensor(residual, "Residual after embedding");

  for (size_t layer = 0; layer < n_layers_; layer++) {
    // debugPrintTensor(residual, "Residual before RMSNorm (att) - layer " +
    //  std::to_string(layer));

    // 打印 RMSNorm 权重 tensor 的前10个元素
    // debugPrintTensor(params_.at("rms_att_w" +
    // std::to_string(layer)),
    //  "RMSNorm weight (att) - layer " + std::to_string(layer));

    // 调用 CUDA 版本的 RMSNorm
    OP::rms_norm(&hidden_states, &residual,
                 &params_.at("rms_att_w" + std::to_string(layer)), eps_);

    // 打印 RMSNorm 后的输出 hidden_states 的前10个元素
    // debugPrintTensor(
    // hidden_states,
    // "Hidden states after RMSNorm (att) - layer " + std::to_string(layer));

    Tensor<float> wq = params_.at("wq" + std::to_string(layer));
    Tensor<float> wk = params_.at("wk" + std::to_string(layer));
    Tensor<float> wv = params_.at("wv" + std::to_string(layer));

    Tensor<float> q_buf, k_buf, v_buf;

    thread_pool.enqueueTask(std::make_shared<OpTask>(
        [&]() { q_buf = avx_OP::matmul(hidden_states, wq); }));
    thread_pool.enqueueTask(std::make_shared<OpTask>(
        [&]() { k_buf = avx_OP::matmul(hidden_states, wk); }));
    thread_pool.enqueueTask(std::make_shared<OpTask>(
        [&]() { v_buf = avx_OP::matmul(hidden_states, wv); }));

    thread_pool.waitForAllTasks();  // Wait for all tasks to finish

    // debugPrintTensor(q_buf, "Q buffer - layer " + std::to_string(layer));
    // debugPrintTensor(k_buf, "K buffer - layer " + std::to_string(layer));
    // debugPrintTensor(v_buf, "V buffer - layer " + std::to_string(layer));

    Tensor<float> q_buf_view = q_buf.view({seq_len, n_q_h_, dqkv_});
    Tensor<float> k_buf_view = k_buf.view({seq_len, n_kv_h_, dqkv_});
    Tensor<float> v_buf_view = v_buf.view({seq_len, n_kv_h_, dqkv_});
    OP::rope(&q_buf_view, current_pos, rope_theta_);
    OP::rope(&k_buf_view, current_pos, rope_theta_);

    // debugPrintTensor(q_buf_view, "Q buffer view after RoPE - layer " +
    //  std::to_string(layer));
    // debugPrintTensor(k_buf_view, "K buffer
    // view after RoPE - layer " +
    //  std::to_string(layer));

    if (kv_cache) {
      kv_cache->k_cache(layer, current_pos) = k_buf_view;
      kv_cache->v_cache(layer, current_pos) = v_buf_view;
    }

    std::vector<float> att_scores(n_q_h_ * kv_cache->size(), 0.0f);
    for (size_t q_head = 0; q_head < n_q_h_; q_head++) {
      size_t kv_head = q_head / n_groups;
      float* q_ptr = q_buf_view.data_ptr() + q_head * dqkv_;
      for (size_t pos = 0; pos < kv_cache->size(); pos++) {
        Tensor<float>& cached_key = kv_cache->k_cache(layer, pos);
        Tensor<float> cached_key_view = cached_key.view({n_kv_h_, dqkv_});
        float* key_ptr = cached_key_view.data_ptr() + kv_head * dqkv_;
        float dot = 0.0f;
        for (size_t d = 0; d < dqkv_; d++) {
          dot += q_ptr[d] * key_ptr[d];
        }
        float scale = 1.0f / sqrtf((float)dqkv_);
        att_scores[q_head * kv_cache->size() + pos] = dot * scale;
      }
    }
    Tensor<float> att_scores_tensor(std::move(att_scores),
                                    {n_q_h_, kv_cache->size()});
    Tensor<float> att_probs = att_scores_tensor;
    OP::softmax(&att_probs, &att_scores_tensor,
                /*dim=*/1, false, n_q_h_);

    // debugPrintTensor(att_scores_tensor,
    //  "Attention scores - layer " +
    //  std::to_string(layer));
    // debugPrintTensor(
    // att_probs, "Attention probabilities -
    // layer " + std::to_string(layer));

    std::vector<float> att_out(n_q_h_ * dqkv_, 0.0f);
    for (size_t q_head = 0; q_head < n_q_h_; q_head++) {
      size_t kv_head = q_head / n_groups;
      for (size_t d = 0; d < dqkv_; d++) {
        float weighted_sum = 0.0f;
        for (size_t pos = 0; pos < kv_cache->size(); pos++) {
          Tensor<float>& cached_val = kv_cache->v_cache(layer, pos);
          Tensor<float> cached_val_view = cached_val.view({n_kv_h_, dqkv_});
          float* val_ptr = cached_val_view.data_ptr() + kv_head * dqkv_;
          weighted_sum +=
              att_probs.data_ptr()[q_head * kv_cache->size() + pos] *
              val_ptr[d];
        }
        att_out[q_head * dqkv_ + d] = weighted_sum;
      }
    }
    Tensor<float> att_heads(std::move(att_out), {1, n_q_h_ * dqkv_});
    // debugPrintTensor(att_heads, "att_heads -
    // layer " + std::to_string(layer));
    Tensor<float> wo = params_.at("wo" + std::to_string(layer));
    Tensor<float> att_proj = avx_OP::matmul(att_heads, wo);

    // debugPrintTensor(att_proj, "att_out -
    // layer " + std::to_string(layer));
    // debugPrintTensor(residual, "residual -
    // layer " + std::to_string(layer));
    residual = residual + att_proj;

    OP::rms_norm(&hidden_states, &residual,
                 &params_.at("rms_ffn_w" + std::to_string(layer)), eps_);

    // FFN计算：分别计算 gate 和 up 分支
    Tensor<float> w_gate = params_.at("w_gate" + std::to_string(layer));
    Tensor<float> w_up = params_.at("w_up" + std::to_string(layer));
    Tensor<float> gate_buf = avx_OP::matmul(hidden_states, w_gate);
    Tensor<float> up_buf = avx_OP::matmul(hidden_states, w_up);

    // print_tensor_shape(
    //     "Gate buffer before SiLU - layer " +
    //     std::to_string(layer), gate_buf);
    OP::silu(&gate_buf, &gate_buf);

    OP::multiply(&gate_buf, &gate_buf, &up_buf);

    // FFN下投影及残差连接
    Tensor<float> w_down = params_.at("w_down" + std::to_string(layer));
    Tensor<float> ffn_out = avx_OP::matmul(gate_buf, w_down);
    residual = residual + ffn_out;
  }

  // 最后一步：整体 RMSNorm 和 LM Head 计算 logits
  OP::rms_norm(&hidden_states, &residual, &params_.at("rms_out_w"), eps_);

  std::vector<float> last_hidden_data(d_, 0.0f);
  Tensor<float> last_hidden(std::move(last_hidden_data), {1, d_});
  size_t start = (seq_len - 1) * d_;
  std::copy(hidden_states.data_ptr() + start,
            hidden_states.data_ptr() + start + d_, last_hidden.data_ptr());

  Tensor<float> lm_head = params_.at("lm_head");
  Tensor<float> logits = avx_OP::matmul(last_hidden, lm_head);

  // debugPrintTensor(logits, "CPU final logits");

  return logits;
}

Tensor<float> LlamaModel::forward_cuda(const Tensor<uint32_t>* input, KVCache* kv_cache) {
  if (compute_precision_ == PrecisionType::FP32) {
    const size_t seq_len = input->numel();
    const size_t current_pos = kv_cache ? kv_cache->size() - seq_len : 0;
    const size_t n_groups = n_q_h_ / n_kv_h_;

    Tensor<float> residual({seq_len, d_}, Device::CUDA);
    Tensor<float> hidden_states({seq_len, d_}, Device::CUDA);

    // Print input tensor details
    // debugPrintTensor(*input, "Input tensor");

    cuda_OP::gather(&residual, input, &params_.at("embedding_table"));

    // Debug print residual after embedding
    // debugPrintTensor(residual, "Residual after embedding");

    for (size_t layer = 0; layer < n_layers_; layer++) {
      // debugPrintTensor(residual, "Residual before RMSNorm (att) - layer " +
      //  std::to_string(layer));

      // 打印 RMSNorm 权重 tensor 的前10个元素
      // debugPrintTensor(params_.at("rms_att_w" +
      // std::to_string(layer)),
      //  "RMSNorm weight (att) - layer " + std::to_string(layer));

      // 调用 CUDA 版本的 RMSNorm
      cuda_OP::rms_norm(&hidden_states, &residual,
                        &params_.at("rms_att_w" + std::to_string(layer)), eps_);

      // 打印 RMSNorm 后的输出 hidden_states 的前10个元素
      // debugPrintTensor(
      // hidden_states,
      // "Hidden states after RMSNorm (att) - layer " +
      // std::to_string(layer));

      Tensor<float>& wq = params_.at("wq" + std::to_string(layer));
      Tensor<float>& wk = params_.at("wk" + std::to_string(layer));
      Tensor<float>& wv = params_.at("wv" + std::to_string(layer));

      if (wq.device() != Device::CUDA || wk.device() != Device::CUDA ||
          wv.device() != Device::CUDA) {
        throw std::runtime_error("QKV weights must be on CUDA device");
      }

      cudaStream_t streams[3];
      for (int i = 0; i < 3; i++) {
        cudaError_t err = cudaStreamCreate(&streams[i]);
        if (err != cudaSuccess) {
          throw std::runtime_error("Failed to create CUDA stream");
        }
      }

      // Asynchronous kernel calls, use different streams for
      // parallel execution
      Tensor<float> q_buf = cuda_OP::matmul(hidden_states, wq, streams[0]);
      Tensor<float> k_buf = cuda_OP::matmul(hidden_states, wk, streams[1]);
      Tensor<float> v_buf = cuda_OP::matmul(hidden_states, wv, streams[2]);

      for (int i = 0; i < 3; i++) {
        cudaError_t err = cudaStreamSynchronize(streams[i]);
        if (err != cudaSuccess) {
          throw std::runtime_error("Stream synchronization failed");
        }
        cudaStreamDestroy(streams[i]);
      }

      // Debug print QKV tensors
      // debugPrintTensor(q_buf, "Q buffer - layer " +
      // std::to_string(layer)); debugPrintTensor(k_buf, "K
      // buffer - layer " + std::to_string(layer));
      // debugPrintTensor(v_buf, "V buffer - layer " +
      // std::to_string(layer));

      Tensor<float> q_buf_view = q_buf.view({seq_len, n_q_h_, dqkv_});
      Tensor<float> k_buf_view = k_buf.view({seq_len, n_kv_h_, dqkv_});
      Tensor<float> v_buf_view = v_buf.view({seq_len, n_kv_h_, dqkv_});
      cuda_OP::rope(&q_buf_view, current_pos, rope_theta_);
      cuda_OP::rope(&k_buf_view, current_pos, rope_theta_);

      if (kv_cache) {
        size_t row_size = n_kv_h_ * dqkv_;
        for (size_t i = 0; i < seq_len; i++) {
          Tensor<float> k_i({1, row_size}, Device::CUDA);
          Tensor<float> v_i({1, row_size}, Device::CUDA);

          cudaError_t err =
              cudaMemcpy(k_i.data_ptr(), k_buf_view.data_ptr() + i * row_size,
                         row_size * sizeof(float), cudaMemcpyDeviceToDevice);
          if (err != cudaSuccess) {
            throw std::runtime_error("KV Cache copy failed");
          }

          err = cudaMemcpy(v_i.data_ptr(), v_buf_view.data_ptr() + i * row_size,
                           row_size * sizeof(float), cudaMemcpyDeviceToDevice);
          if (err != cudaSuccess) {
            throw std::runtime_error("KV Cache copy failed");
          }

          kv_cache->k_cache(layer, current_pos + i) = std::move(k_i);
          kv_cache->v_cache(layer, current_pos + i) = std::move(v_i);
        }
      }

      // Debug print tensors after RoPE
      // debugPrintTensor(q_buf_view, "Q buffer view after RoPE -
      // layer " +
      //  std::to_string(layer));
      // debugPrintTensor(k_buf_view, "K buffer
      // view after RoPE - layer " +
      //  std::to_string(layer));

      Tensor<float> Q_3d = q_buf_view;
      Tensor<float> total_K, total_V;
      size_t total_seq_len = seq_len;

      if (current_pos != 0) {
        size_t cached_len = current_pos;
        total_seq_len = cached_len + seq_len;
        size_t row_size = n_kv_h_ * dqkv_;

        total_K = Tensor<float>({total_seq_len, n_kv_h_, dqkv_}, Device::CUDA);
        total_V = Tensor<float>({total_seq_len, n_kv_h_, dqkv_}, Device::CUDA);

        for (size_t pos = 0; pos < cached_len; pos++) {
          Tensor<float>& cached_k = kv_cache->k_cache(layer, pos);
          Tensor<float>& cached_v = kv_cache->v_cache(layer, pos);

          cudaError_t err =
              cudaMemcpy(total_K.data_ptr() + pos * row_size, cached_k.data_ptr(),
                         row_size * sizeof(float), cudaMemcpyDeviceToDevice);
          if (err != cudaSuccess) {
            throw std::runtime_error("Cache copy failed");
          }

          err =
              cudaMemcpy(total_V.data_ptr() + pos * row_size, cached_v.data_ptr(),
                         row_size * sizeof(float), cudaMemcpyDeviceToDevice);
          if (err != cudaSuccess) {
            throw std::runtime_error("Cache copy failed");
          }
        }

        cudaError_t err = cudaMemcpy(
            total_K.data_ptr() + cached_len * row_size, k_buf_view.data_ptr(),
            seq_len * row_size * sizeof(float), cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
          throw std::runtime_error("Current K copy failed");
        }

        err = cudaMemcpy(
            total_V.data_ptr() + cached_len * row_size, v_buf_view.data_ptr(),
            seq_len * row_size * sizeof(float), cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
          throw std::runtime_error("Current V copy failed");
        }
      } else {
        total_K = k_buf_view;
        total_V = v_buf_view;
      }

      Tensor<float> att_scores({n_q_h_, total_seq_len}, Device::CUDA);
      cuda_OP::compute_attention_scores(Q_3d, total_K, n_q_h_, dqkv_, att_scores,
                                        n_kv_h_);

      cuda_OP::softmax(&att_scores, &att_scores,
                       /*dim=*/1, false, current_pos);

      // Debug print attention scores
      // debugPrintTensor(att_scores,
      //  "Attention scores - layer " +
      //  std::to_string(layer));

      Tensor<float> att_heads({n_q_h_, dqkv_}, Device::CUDA);
      cuda_OP::compute_att_output(att_scores, total_V, n_q_h_, dqkv_, att_heads,
                                  n_kv_h_);
      // debugPrintTensor(att_heads, "att_heads -
      // layer " + std::to_string(layer));

      // Tensor<float> att_out({1, d_},
      // Device::CUDA);
      Tensor<float>& wo = params_.at("wo" + std::to_string(layer));

      Tensor<float> att_proj =
          cuda_OP::matmul(att_heads.view({1, n_q_h_ * dqkv_}), wo);

      // residual = residual + att_proj;

      // debugPrintTensor(att_proj, "att_out -
      // layer " + std::to_string(layer));
      // debugPrintTensor(residual, "residual -
      // layer " + std::to_string(layer));
      residual = residual + att_proj;
      // (3) FFN 前的 RMSNorm
      cuda_OP::rms_norm(&hidden_states, &residual,
                        &params_.at("rms_ffn_w" + std::to_string(layer)), eps_);
      // //debugPrintTensor(hidden_states, "CUDA
      // hidden_states after RMSNorm(ffn)");

      // FFN
      Tensor<float>& w_gate = params_.at("w_gate" + std::to_string(layer));
      Tensor<float>& w_up = params_.at("w_up" + std::to_string(layer));
      Tensor<float>& w_down = params_.at("w_down" + std::to_string(layer));

      Tensor<float> gate_buf = cuda_OP::matmul(hidden_states, w_gate);
      Tensor<float> up_buf = cuda_OP::matmul(hidden_states, w_up);

      cuda_OP::silu(&gate_buf, &gate_buf);
      cuda_OP::multiply(&gate_buf, &gate_buf, &up_buf);

      Tensor<float> ffn_out = cuda_OP::matmul(gate_buf, w_down);
      residual = residual + ffn_out;
      // //debugPrintTensor(residual, "CUDA
      // residual after FFN");
    }

    // 5. 最终 RMSNorm + lm_head
    Tensor<float> final_h({seq_len, d_}, Device::CUDA);
    cuda_OP::rms_norm(&final_h, &residual, &params_.at("rms_out_w"), eps_);
    // //debugPrintTensor(final_h, "CUDA final_h after final RMSNorm");

    Tensor<float>& lm_head = params_.at("lm_head");
    size_t vocab_size = lm_head.sizes()[1];
    Tensor<float> logits({seq_len, vocab_size}, Device::CUDA);
    logits = cuda_OP::matmul(final_h, lm_head);
    return logits.cpu();
  } else if (compute_precision_ == PrecisionType::FP16) {
    // 实现 FP16 前向传播
    // 1. 将输入转换为 FP16
    // 2. 使用 FP16 权重和计算
    // 3. 将输出转换回 FP32
    
    // 临时返回空结果，后续实现
    return Tensor<float>({1, vocab_size_});
  } else if (compute_precision_ == PrecisionType::BF16) {
    // 实现 BF16 前向传播
    
    // 临时返回空结果，后续实现
    return Tensor<float>({1, vocab_size_});
  }
  
  // 默认返回空结果
  return Tensor<float>({1, vocab_size_});
}

Tensor<float> LlamaModel::forward(const Tensor<uint32_t>* input,
                                  ThreadPool& thread_pool, KVCache* kv_cache) {
  if (input->device() == Device::CPU) {
    return forward_cpu(input, thread_pool, kv_cache);
  } else if (input->device() == Device::CUDA) {
    return forward_cuda(input, kv_cache);
  } else {
    throw std::runtime_error("Unsupported device for input tensor in prefill");
  }
}