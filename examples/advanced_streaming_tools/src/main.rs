use merco_agents::{
    Agent, AgentModelConfig, LlmConfig, Provider, StreamingHandler, StreamingChunk, StreamingResponse,
    Task, AgentRole, AgentCapabilities,
};
use futures::StreamExt;
use std::io::Write;
use dotenv::dotenv;
use colored::*;
use merco_llmproxy::{merco_tool, get_all_tools};
use serde_json::json;

// Advanced streaming handler with colored output and detailed logging
struct AdvancedStreamingHandler {
    chunk_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    tool_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl AdvancedStreamingHandler {
    fn new() -> Self {
        Self {
            chunk_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            tool_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }
}

impl StreamingHandler for AdvancedStreamingHandler {
    fn handle_chunk(&self, chunk: StreamingChunk) {
        let chunk_num = self.chunk_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        
        if !chunk.content.is_empty() {
            // Color code different types of content
            let colored_content = if chunk.content.contains("🔧") {
                chunk.content.cyan()
            } else if chunk.content.contains("✅") {
                chunk.content.green()
            } else if chunk.content.contains("⏳") {
                chunk.content.yellow()
            } else {
                chunk.content.white()
            };
            
            print!("{}", colored_content);
            std::io::stdout().flush().unwrap();
        }
        
        if chunk.has_tool_calls {
            println!("\n{}", "🔧 Tool calls detected in stream!".bright_cyan());
        }
        
        // Show metadata if available
        if !chunk.metadata.is_empty() {
            println!("\n{}", format!("📊 Metadata: {:?}", chunk.metadata).dimmed());
        }
    }
    
    fn handle_tool_calls(&self, tool_calls: Vec<merco_agents::agent::ToolCall>) {
        let tool_num = self.tool_count.fetch_add(tool_calls.len(), std::sync::atomic::Ordering::Relaxed);
        
        println!("\n{}", "🛠️ Tool Calls Executed:".bright_magenta());
        for (i, call) in tool_calls.iter().enumerate() {
            let tool_index = tool_num + i + 1;
            println!("  {}. {} - {}", 
                tool_index.to_string().bright_blue(), 
                call.tool_name.bright_cyan(), 
                call.parameters.bright_white()
            );
            println!("     {} {}", "Result:".bright_green(), call.result.bright_white());
            println!("     {} {}ms", "Execution time:".dimmed(), call.execution_time_ms.to_string().bright_yellow());
        }
    }
    
    fn handle_final(&self, response: StreamingResponse) {
        println!("\n{}", "=".repeat(60).bright_blue());
        println!("{}", "📊 Streaming Complete - Summary".bright_green());
        println!("{}", "=".repeat(60).bright_blue());
        println!("{} {}", "Total tokens:".bright_white(), response.total_tokens.to_string().bright_yellow());
        println!("{} {}ms", "Execution time:".bright_white(), response.execution_time_ms.to_string().bright_yellow());
        println!("{} {}", "Model:".bright_white(), response.model_used.bright_cyan());
        println!("{} {}", "Success:".bright_white(), if response.success { "✅".bright_green() } else { "❌".bright_red() });
        
        if !response.tools_used.is_empty() {
            println!("{} {:?}", "Tools used:".bright_white(), response.tools_used.join(", ").bright_cyan());
        }
        
        if let Some(error) = response.error {
            println!("{} {}", "Error:".bright_red(), error.bright_red());
        }
        
        println!("{}", "=".repeat(60).bright_blue());
    }
    
    fn handle_error(&self, error: String) {
        eprintln!("{} {}", "❌ Streaming Error:".bright_red(), error.bright_red());
    }
}

#[merco_tool(description = "Get the current weather for a location")]
fn get_weather(location: String, unit: Option<String>) -> String {
    let unit = unit.unwrap_or_else(|| "celsius".to_string());
    format!("The weather in {} is 22°{}", location, if unit == "fahrenheit" { "F" } else { "C" })
}

#[merco_tool(description = "Search for information on the web")]
fn web_search(query: String) -> String {
    format!("Search results for '{}': Found 42 relevant articles about this topic.", query)
}

#[merco_tool(description = "Calculate a mathematical expression")]
fn calculate(expression: String) -> String {
    // Simple mock calculation
    format!("Result of '{}' = 42", expression)
}



#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables
    dotenv().ok();

    // Check for API key
    let api_key = std::env::var("OPENROUTER_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .unwrap_or_else(|_| {
            println!("No API key found. Please set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable.");
            std::process::exit(1);
        });

    // Create agent configuration
    let llm_config = LlmConfig::new_with_base_url(
        Provider::OpenAI,
        Some(api_key),
        "https://openrouter.ai/api/v1".to_string()
    );
    
    let config = AgentModelConfig::new(
        llm_config,
        "openai/gpt-4o-mini".to_string(),
        0.7,
        1500
    );

    // Get available tools
    let tools = get_all_tools();
    
    // Create agent
    let mut agent = Agent::new(
        "Advanced Streaming Agent".to_string(),
        "An advanced streaming agent for testing complex tool calls".to_string(),
        AgentRole::new("Assistant".to_string(), "A helpful AI assistant with advanced capabilities".to_string()),
        config,
        tools, // Now with tools!
        AgentCapabilities {
            max_concurrent_tasks: 3,
            supported_output_formats: vec![merco_agents::agent::role::OutputFormat::Text],
        },
    );
    
    println!("{}", "🚀 Advanced Streaming Tool Call Example".bright_green());
    println!("{}", "=".repeat(50).bright_blue());
    
    // Test 1: Complex multi-tool scenario
    println!("\n{}", "Test 1: Complex Multi-Tool Scenario".bright_yellow());
    println!("{}", "-".repeat(40));
    let task1 = Task::new(
        "I need to plan a trip to Japan. Please get the weather in Tokyo, search for information about cherry blossoms, and calculate the cost for a 7-day trip. Use all available tools to help me plan.".to_string(),
        None,
    );
    
    let handler = AdvancedStreamingHandler::new();
    let mut stream = agent.call_stream_with_handler(task1, handler).await;
    
    println!("{}", "Streaming response:".bright_white());
    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                if chunk.is_final {
                    break;
                }
            }
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
        }
    }
    
    println!("\n{}", "=".repeat(50).bright_blue());
    
    // Test 2: Sequential tool calls
    println!("\n{}", "Test 2: Sequential Tool Calls".bright_yellow());
    println!("{}", "-".repeat(40));
    let task2 = Task::new(
        "First, get the weather in London. Then, based on that weather, search for appropriate activities. Finally, calculate the total cost for a weekend trip.".to_string(),
        None,
    );
    
    let handler2 = AdvancedStreamingHandler::new();
    let mut stream2 = agent.call_stream_with_handler(task2, handler2).await;
    
    println!("{}", "Streaming response:".bright_white());
    while let Some(chunk_result) = stream2.next().await {
        match chunk_result {
            Ok(chunk) => {
                if chunk.is_final {
                    break;
                }
            }
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
        }
    }
    
    println!("\n{}", "✅ All advanced tests completed!".bright_green());
    
    Ok(())
}
