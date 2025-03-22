mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use tokenizers::Tokenizer;
use std::io::{self, Write};
use crate::model::ChatMessage;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() > 1 && args[1] == "chat" {
        chat_mode();
    } else {
        story_mode();
    }
}

fn story_mode() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = "Once upon a time";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    print!("\n{}", input);
    let output_ids = llama.generate(
        input_ids,
        500,
        0.8,
        30,
        1.,
    );
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}

fn chat_mode() {
    println!("欢迎使用AI聊天助手！输入'exit'退出对话。");
    
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    
    // 初始化对话历史
    let mut messages = Vec::new();
    
    // 添加系统消息
    messages.push(ChatMessage {
        role: "system".to_string(),
        content: "你是一个友好的AI助手，能够回答用户的各种问题。".to_string(),
    });
    
    loop {
        print!("\n用户: ");
        io::stdout().flush().unwrap();
        
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input).unwrap();
        let user_input = user_input.trim();
        
        if user_input.to_lowercase() == "exit" {
            println!("再见！");
            break;
        }
        
        // 添加用户消息
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: user_input.to_string(),
        });
        
        // 生成AI回复
        print!("AI助手: ");
        io::stdout().flush().unwrap();
        
        // 构建输入文本 - 简化为故事续写模式
        let mut prompt = String::new();
        prompt.push_str(&format!("用户问题: {}\n\n助手回答: ", user_input));
        
        let binding = tokenizer.encode(prompt.as_str(), true).unwrap();
        let input_ids = binding.get_ids();
        
        let output_ids = llama.generate(
            input_ids,
            200,  // max_len
            0.8,  // top_p
            30,   // top_k
            0.7,  // temperature
        );
        
        let response = tokenizer.decode(&output_ids, true).unwrap();
        // 只保留助手回答部分
        let ai_response = response.replace(&prompt, "");
        
        println!("{}", ai_response);
        
        // 添加AI回复到历史
        messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: ai_response,
        });
    }
}
