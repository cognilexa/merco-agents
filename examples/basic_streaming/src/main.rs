use merco_agents::{
    Agent, AgentModelConfig, AgentRole, AgentCapabilities, OutputFormat,
    LlmConfig, Provider,
    StreamingHandler, StreamingChunk, StreamingResponse,
};
use std::io::Write;

// Simple streaming handler
struct BasicStreamingHandler;

impl StreamingHandler for BasicStreamingHandler {
    fn handle_chunk(&self, chunk: StreamingChunk) {
        print!("{}", chunk.content);
        std::io::stdout().flush().unwrap();
    }
    
    fn handle_final(&self, response: StreamingResponse) {
        println!("\n\n=== Streaming Complete ===");
        println!("Total tokens: {}", response.total_tokens);
        println!("Execution time: {}ms", response.execution_time_ms);
        println!("Model: {}", response.model_used);
    }
    
    fn handle_error(&self, error: String) {
        eprintln!("❌ Error: {}", error);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables
    dotenv::dotenv().ok();
    
    // Get API key from environment
    let api_key = std::env::var("OPENROUTER_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .unwrap_or_else(|_| {
            println!("❌ No API key found. Please set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable.");
            std::process::exit(1);
        });
    
    println!("🚀 Basic Streaming Agent Example");
    println!("=================================");
    
    // Create LLM configuration
    let llm_config = LlmConfig::new_with_base_url(
        Provider::OpenAI,
        Some(api_key),
        "https://openrouter.ai/api/v1".to_string(),
    );
    
    let agent_llm_config = AgentModelConfig::new(
        llm_config,
        "openai/gpt-4o-mini".to_string(),
        0.7,
        1000,
    );
    
    // Create agent
    let role = AgentRole::new(
        "Helpful Assistant".to_string(),
        "You are a helpful AI assistant that provides clear and informative responses.".to_string(),
    );
    
    let capabilities = AgentCapabilities {
        max_concurrent_tasks: 1,
        supported_output_formats: vec![OutputFormat::Text],
    };
    
    let mut agent = Agent::new(
        "Streaming Assistant".to_string(),
        "An AI assistant that streams responses in real-time".to_string(),
        role,
        agent_llm_config,
        vec![], // No tools for this example
        capabilities,
    );
    
    println!("✅ Agent created successfully");
    println!();
    
    // Example 1: Basic streaming
    println!("📝 Example 1: Basic Streaming");
    println!("-----------------------------");
    let prompt1 = "Write a short story about a robot learning to paint. Make it about 100 words.";
    
    match agent.call_str_stream(prompt1).await {
        Ok(response) => {
            println!("\n✅ Basic streaming completed");
            println!("Final content length: {} characters", response.content.len());
        }
        Err(e) => {
            eprintln!("❌ Basic streaming failed: {}", e);
        }
    }
    
    println!("\n{}", "=".repeat(50));
    
    // Example 2: Streaming with custom handler
    println!("📝 Example 2: Streaming with Custom Handler");
    println!("-------------------------------------------");
    let prompt2 = "Explain how streaming works in AI systems. Keep it concise.";
    
    let handler = BasicStreamingHandler;
    
    match agent.call_str_stream_with_handler(prompt2, handler).await {
        Ok(response) => {
            println!("\n✅ Custom handler streaming completed");
        }
        Err(e) => {
            eprintln!("❌ Custom handler streaming failed: {}", e);
        }
    }
    
    println!("\n🎉 All streaming examples completed!");
    
    Ok(())
}
