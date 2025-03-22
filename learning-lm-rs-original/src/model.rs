use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );

            OP::matmul_transb(
                &mut residual,
                1.0,
                &hidden_states,
                &self.params.wo[layer],
                1.0
            );

            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        // 初始化结果，复制输入的token_ids
        let mut result = token_ids.to_vec();
        
        // 创建一个新的KV缓存
        let mut cache = self.new_cache();
        
        // 处理初始输入序列
        let input_tensor = Tensor::new(token_ids.to_vec(), &vec![token_ids.len()]);
        let mut logits = self.forward(&input_tensor, &mut cache);
        
        // 生成新的token，直到达到最大长度或生成了结束符
        for _ in 0..max_len {
            // 从logits中采样下一个token
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);
            
            // 将新token添加到结果中
            result.push(next_token);
            
            // 如果生成了结束符，停止生成
            if next_token == self.eos_token_id {
                break;
            }
            
            // 使用新token作为下一轮的输入
            let next_input = Tensor::new(vec![next_token], &vec![1]);
            logits = self.forward(&next_input, &mut cache);
        }
        
        result
    }

    pub fn chat(&self, messages: &[ChatMessage], max_len: usize, top_p: f32, top_k: u32, temperature: f32) -> String {
        // 创建一个新的KV缓存
        let _cache = self.new_cache();
        
        // 构建输入文本 - 简化为故事续写模式
        let mut prompt = String::new();
        
        // 创建一个默认消息，避免临时值问题
        let default_msg = ChatMessage {
            role: "user".to_string(),
            content: "你好".to_string(),
        };
        
        // 获取最后一条用户消息
        let last_user_msg = messages.iter()
            .filter(|msg| msg.role == "user")
            .last()
            .unwrap_or(&default_msg);
        
        prompt.push_str(&format!("用户问题: {}\n\n助手回答: ", last_user_msg.content));
        
        // 使用tokenizer将输入文本转换为token ids
        let tokenizer = tokenizers::Tokenizer::from_file(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("models")
                .join("story")
                .join("tokenizer.json")
        ).unwrap();
        
        let binding = tokenizer.encode(prompt.as_str(), true).unwrap();
        let input_ids = binding.get_ids();
        
        // 生成回复
        let output_ids = self.generate(
            input_ids,
            max_len,
            top_p,
            top_k,
            temperature,
        );
        
        // 解码生成的token
        let response = tokenizer.decode(&output_ids, true).unwrap();
        
        // 只保留助手回答部分
        response.replace(&prompt, "")
    }
}

fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    // 1. 计算注意力分数: Q·K^T / sqrt(dqkv)
    let scale = (dqkv as f32).sqrt();
    let q_data = q.data();
    let k_data = k.data();
    let v_data = v.data();
    let att_data = unsafe { att_scores.data_mut() };
    
    // 对每个KV头和每个组计算注意力分数
    for h in 0..n_kv_h {
        for g in 0..n_groups {
            for i in 0..seq_len {
                for j in 0..total_seq_len {
                    let mut score = 0.0;
                    // 计算点积
                    for d in 0..dqkv {
                        let q_idx = i * n_kv_h * n_groups * dqkv + (h * n_groups + g) * dqkv + d;
                        let k_idx = j * n_kv_h * dqkv + h * dqkv + d;
                        score += q_data[q_idx] * k_data[k_idx];
                    }
                    // 缩放并存储分数
                    let att_idx = h * n_groups * seq_len * total_seq_len + g * seq_len * total_seq_len + i * total_seq_len + j;
                    att_data[att_idx] = score / scale;
                }
            }
        }
    }
    
    // 2. 应用softmax
    OP::masked_softmax(att_scores);
    
    // 3. 计算加权和: attn·V
    let att_data = att_scores.data();
    let hidden_data = unsafe { hidden_states.data_mut() };
    
    // 初始化hidden_states为0
    // 先计算大小，避免同时借用
    let hidden_size = seq_len * n_kv_h * n_groups * dqkv;
    for i in 0..hidden_size {
        hidden_data[i] = 0.0;
    }
    
    // 对每个KV头和每个组计算加权和
    for h in 0..n_kv_h {
        for g in 0..n_groups {
            for i in 0..seq_len {
                for j in 0..total_seq_len {
                    let att_idx = h * n_groups * seq_len * total_seq_len + g * seq_len * total_seq_len + i * total_seq_len + j;
                    let att_val = att_data[att_idx];
                    
                    // 如果注意力权重为0，跳过计算
                    if att_val == 0.0 {
                        continue;
                    }
                    
                    for d in 0..dqkv {
                        let hidden_idx = i * n_kv_h * n_groups * dqkv + (h * n_groups + g) * dqkv + d;
                        let v_idx = j * n_kv_h * dqkv + h * dqkv + d;
                        hidden_data[hidden_idx] += att_val * v_data[v_idx];
                    }
                }
            }
        }
    }
}

fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    // 1. hidden = rms_norm(residual)
    OP::rms_norm(hidden_states, residual, rms_w, eps);
    
    // 2. gate = hidden @ gate_weight.T
    OP::matmul_transb(gate, 0.0, hidden_states, w_gate, 1.0);
    
    // 3. up = hidden @ up_weight.T
    OP::matmul_transb(up, 0.0, hidden_states, w_up, 1.0);
    
    // 4. act = gate * sigmoid(gate) * up (SwiGLU)
    OP::swiglu(up, gate);
    
    // 5. output = act @ down_weight.T
    // 6. residual = output + residual (使用beta=1.0来实现残差连接)
    OP::matmul_transb(residual, 1.0, up, w_down, 1.0);
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use std::path::PathBuf;
    use crate::tensor::float_eq;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
    assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
    assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
    assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
    assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
    assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
    assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
    assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
    assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

}

// 定义聊天消息结构
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}
