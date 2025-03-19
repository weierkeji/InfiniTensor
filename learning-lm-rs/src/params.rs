use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // 首先打印所有张量的名称
        println!("Available tensors:");
        for name in safetensor.names() {
            println!("{}", name);
        }
        
        // 创建一个闭包来从safetensors中获取张量
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor = safetensor.tensor(name).unwrap();
            let shape = tensor.shape().to_vec();
            let data = tensor.data();  // 直接使用data，不需要unwrap
            let data: Vec<f32> = match tensor.dtype() {
                safetensors::Dtype::F32 => data.chunks(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
                _ => panic!("Unsupported dtype"),
            };
            Tensor::new(data, &shape)
        };
        
        // 获取层数
        let n_layers = config.num_hidden_layers;
        
        // 创建存储每层参数的向量
        let mut rms_att_w = Vec::with_capacity(n_layers);
        let mut wq = Vec::with_capacity(n_layers);
        let mut wk = Vec::with_capacity(n_layers);
        let mut wv = Vec::with_capacity(n_layers);
        let mut wo = Vec::with_capacity(n_layers);
        let mut rms_ffn_w = Vec::with_capacity(n_layers);
        let mut w_up = Vec::with_capacity(n_layers);
        let mut w_gate = Vec::with_capacity(n_layers);
        let mut w_down = Vec::with_capacity(n_layers);
        
        // 加载每层的参数
        for i in 0..n_layers {
            rms_att_w.push(get_tensor(&format!("model.layers.{}.input_layernorm.weight", i)));
            wq.push(get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i)));
            wk.push(get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", i)));
            wv.push(get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", i)));
            wo.push(get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", i)));
            rms_ffn_w.push(get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", i)));
            w_up.push(get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", i)));
            w_gate.push(get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", i)));
            w_down.push(get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", i)));
        }
        
        // 加载embedding和输出层参数
        let embedding_table = get_tensor("lm_head.weight");
        let rms_out_w = get_tensor("model.norm.weight");
        // 当tie_word_embeddings为true时，lm_head使用相同的embedding权重
        let lm_head = if config.tie_word_embeddings {
            embedding_table.clone()  // 现在可以安全地克隆了
        } else {
            get_tensor("lm_head.weight")
        };
        
        LLamaParams {
            embedding_table,
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w,
            lm_head,
        }
    }
}
