use merco_agents::{
    Agent, AgentModelConfig, AgentRole, AgentCapabilities, OutputFormat,
    LlmConfig, Provider,
    StreamingHandler, StreamingChunk, StreamingResponse,
};
use futures::StreamExt;
use std::io::Write;

// Custom streaming handler that shows real-time streaming with tool call support
struct RealTimeStreamingHandler;

impl StreamingHandler for RealTimeStreamingHandler {
    fn handle_chunk(&self, chunk: StreamingChunk) {
        if !chunk.content.is_empty() {
            print!("{}", chunk.content);
            std::io::stdout().flush().unwrap();
        }
        
        if chunk.has_tool_calls {
            println!("\n🔧 Tool calls detected in stream!");
        }
    }
    
    fn handle_tool_calls(&self, tool_calls: Vec<crate::agent::agent::ToolCall>) {
        println!("\n🛠️ Tool Calls:");
        for (i, call) in tool_calls.iter().enumerate() {
            println!("  {}. {} - {}", i + 1, call.tool_name, call.parameters);
        }
    }
    
    fn handle_final(&self, response: StreamingResponse) {
        println!("\n\n=== Streaming Complete ===");
        println!("Total tokens: {}", response.total_tokens);
        println!("Execution time: {}ms", response.execution_time_ms);
        println!("Model: {}", response.model_used);
        if !response.tools_used.is_empty() {
            println!("Tools used: {:?}", response.tools_used);
        }
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
    
    println!("🚀 Real Streaming Agent Example");
    println!("===============================");
    
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
        "Storyteller".to_string(),
        "You are a creative storyteller who writes engaging stories.".to_string(),
    );
    
    let capabilities = AgentCapabilities {
        max_concurrent_tasks: 1,
        supported_output_formats: vec![OutputFormat::Text],
    };
    
    let mut agent = Agent::new(
        "Storytelling Agent".to_string(),
        "An AI that tells stories with real-time streaming".to_string(),
        role,
        agent_llm_config,
        vec![], // No tools for this example
        capabilities,
    );
    
    println!("✅ Agent created successfully");
    println!();
    
    // Example 1: Basic streaming with default handler
    println!("📝 Example 1: Basic Streaming");
    println!("-----------------------------");
    let prompt1 = "Write a short story about a robot learning to paint. Make it about 200 words.";
    
    // Get the stream
    let mut stream = agent.call_str_stream(prompt1).await;
    
    // Consume chunks as they arrive
    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                print!("{}", chunk.content);
                std::io::stdout().flush().unwrap();
                
                if chunk.is_final {
                    println!("\n✅ Stream complete!");
                    break;
                }
            }
            Err(e) => {
                eprintln!("❌ Error: {}", e);
                break;
            }
        }
    }
    
    println!("\n{}", "=".repeat(50));
    
    // Example 2: Streaming with custom handler
    println!("📝 Example 2: Streaming with Custom Handler");
    println!("-------------------------------------------");
    let prompt2 = "Write a poem about the ocean. Make it 4 stanzas long.";
    
    let handler = RealTimeStreamingHandler;
    let mut stream = agent.call_str_stream_with_handler(prompt2, handler).await;
    
    // Consume chunks as they arrive
    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                if chunk.is_final {
                    println!("\n✅ Custom handler stream complete!");
                    break;
                }
            }
            Err(e) => {
                eprintln!("❌ Error: {}", e);
                break;
            }
        }
    }
    
    println!("\n🎉 All streaming examples completed!");
    println!("\n💡 Key Features Demonstrated:");
    println!("   • Real-time content streaming");
    println!("   • Continuous chunk delivery");
    println!("   • Custom streaming handlers");
    println!("   • Proper stream consumption");
    println!("   • Error handling in streams");
    
    Ok(())
}
