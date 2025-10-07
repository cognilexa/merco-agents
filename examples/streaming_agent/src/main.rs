use merco_agents::{
    Agent, AgentModelConfig, AgentRole, AgentCapabilities, OutputFormat,
    LlmConfig, Provider,
    StreamingHandler, StreamingChunk, StreamingResponse,
};
use std::io::Write;
use std::sync::Arc;
use tokio::sync::Mutex;

// Custom streaming handler that adds timestamps
struct TimestampedStreamingHandler;

impl StreamingHandler for TimestampedStreamingHandler {
    fn handle_chunk(&self, chunk: StreamingChunk) {
        let timestamp = chrono::Utc::now().format("%H:%M:%S%.3f");
        print!("[{}] {}", timestamp, chunk.content);
        std::io::stdout().flush().unwrap();
    }
    
    fn handle_final(&self, response: StreamingResponse) {
        println!("\n\n=== Streaming Completed ===");
        println!("Total tokens: {}", response.total_tokens);
        println!("Execution time: {}ms", response.execution_time_ms);
        println!("Model: {}", response.model_used);
        println!("Temperature: {}", response.temperature);
        if !response.tools_used.is_empty() {
            println!("Tools used: {:?}", response.tools_used);
        }
    }
    
    fn handle_error(&self, error: String) {
        eprintln!("❌ Streaming error: {}", error);
    }
}

// Custom streaming handler that collects content and provides statistics
struct CollectingStreamingHandler {
    content: Arc<Mutex<String>>,
    chunk_count: Arc<Mutex<usize>>,
}

impl CollectingStreamingHandler {
    fn new() -> Self {
        Self {
            content: Arc::new(Mutex::new(String::new())),
            chunk_count: Arc::new(Mutex::new(0)),
        }
    }
    
    async fn get_stats(&self) -> (String, usize) {
        let content = self.content.lock().await;
        let count = self.chunk_count.lock().await;
        (content.clone(), *count)
    }
}

impl StreamingHandler for CollectingStreamingHandler {
    fn handle_chunk(&self, chunk: StreamingChunk) {
        print!("{}", chunk.content);
        std::io::stdout().flush().unwrap();
        
        // In a real async context, you'd use tokio::spawn or similar
        // For this example, we'll just print the chunk info
        println!("[CHUNK] Length: {}, Final: {}", chunk.content.len(), chunk.is_final);
    }
    
    fn handle_final(&self, response: StreamingResponse) {
        println!("\n\n=== Streaming Statistics ===");
        println!("Total tokens: {}", response.total_tokens);
        println!("Execution time: {}ms", response.execution_time_ms);
        println!("Model: {}", response.model_used);
        println!("Temperature: {}", response.temperature);
        println!("Content length: {} characters", response.content.len());
        if !response.tools_used.is_empty() {
            println!("Tools used: {:?}", response.tools_used);
        }
    }
    
    fn handle_error(&self, error: String) {
        eprintln!("❌ Collecting streaming error: {}", error);
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
            println!("No API key found. Please set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable.");
            std::process::exit(1);
        });
    
    println!("🚀 Streaming Agent Example");
    println!("=========================");
    
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
        2000,
    );
    
    // Create agent
    let role = AgentRole::new(
        "Creative Writer".to_string(),
        "You are a creative writer who tells engaging stories. Write in a flowing, narrative style.".to_string(),
    );
    
    let capabilities = AgentCapabilities {
        max_concurrent_tasks: 1,
        supported_output_formats: vec![OutputFormat::Text],
    };
    
    let mut agent = Agent::new(
        "Storyteller".to_string(),
        "A creative AI that writes engaging stories".to_string(),
        role,
        agent_llm_config,
        vec![], // No tools for this example
        capabilities,
    );
    
    println!("✅ Agent created successfully");
    println!();
    
    // Example 1: Basic streaming with default handler
    println!("📝 Example 1: Basic Streaming");
    println!("------------------------------");
    let prompt1 = "Write a short story about a robot learning to paint. Make it about 200 words.";
    
    match agent.call_str_stream(prompt1).await {
        Ok(response) => {
            println!("\n✅ Streaming completed successfully");
            println!("Final content length: {} characters", response.content.len());
        }
        Err(e) => {
            eprintln!("❌ Streaming failed: {}", e);
        }
    }
    
    println!("\n{}", "=".repeat(50));
    
    // Example 2: Streaming with custom timestamped handler
    println!("📝 Example 2: Streaming with Timestamps");
    println!("---------------------------------------");
    let prompt2 = "Write a poem about the ocean. Make it 4 stanzas long.";
    
    let timestamped_handler = TimestampedStreamingHandler;
    
    match agent.call_str_stream_with_handler(prompt2, timestamped_handler).await {
        Ok(response) => {
            println!("\n✅ Timestamped streaming completed");
        }
        Err(e) => {
            eprintln!("❌ Timestamped streaming failed: {}", e);
        }
    }
    
    println!("\n{}", "=".repeat(50));
    
    // Example 3: Streaming with collecting handler
    println!("📝 Example 3: Streaming with Collecting Handler");
    println!("-----------------------------------------------");
    let prompt3 = "Write a technical explanation of how streaming works in AI systems. Make it detailed and informative.";
    
    let collecting_handler = CollectingStreamingHandler::new();
    
    match agent.call_str_stream_with_handler(prompt3, collecting_handler).await {
        Ok(response) => {
            println!("\n✅ Collecting streaming completed");
        }
        Err(e) => {
            eprintln!("❌ Collecting streaming failed: {}", e);
        }
    }
    
    println!("\n🎉 All streaming examples completed!");
    
    Ok(())
}
