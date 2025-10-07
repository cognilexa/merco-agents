use merco_agents::{
    Agent, AgentModelConfig, LlmConfig, Provider, StreamingHandler, StreamingChunk, StreamingResponse,
    Task, AgentRole, AgentCapabilities,
};
use futures::StreamExt;
use std::io::Write;
use dotenv::dotenv;
use merco_llmproxy::{merco_tool, get_all_tools};
use serde_json::json;

// Simple streaming handler that shows real-time tool execution
struct BasicStreamingHandler;

impl StreamingHandler for BasicStreamingHandler {
    fn handle_chunk(&self, chunk: StreamingChunk) {
        if !chunk.content.is_empty() {
            print!("{}", chunk.content);
            std::io::stdout().flush().unwrap();
        }
        
        if chunk.has_tool_calls {
            println!("\n🔧 Tool calls detected in stream!");
        }
    }
    
    fn handle_tool_calls(&self, tool_calls: Vec<merco_agents::agent::ToolCall>) {
        println!("\n🛠️ Tool Calls Executed:");
        for (i, call) in tool_calls.iter().enumerate() {
            println!("  {}. {} - {}", i + 1, call.tool_name, call.parameters);
            println!("     Result: {}", call.result);
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
        1000
    );

    // Get available tools
    let tools = get_all_tools();
    
    // Create agent
    let mut agent = Agent::new(
        "Streaming Agent".to_string(),
        "A streaming agent for testing tool calls".to_string(),
        AgentRole::new("Assistant".to_string(), "A helpful AI assistant".to_string()),
        config,
        tools, // Now with tools!
        AgentCapabilities {
            max_concurrent_tasks: 1,
            supported_output_formats: vec![merco_agents::agent::role::OutputFormat::Text],
        },
    );
    
    println!("🚀 Basic Streaming Tool Call Example");
    println!("=====================================\n");
    
    // Test 1: Simple tool call
    println!("Test 1: Simple tool call");
    println!("------------------------");
    let task1 = Task::new(
        "What's the weather like in Tokyo? Use the get_weather tool.".to_string(),
        None,
    );
    
    let handler = BasicStreamingHandler;
    let mut stream = agent.call_stream_with_handler(task1, handler).await;
    
    println!("Streaming response:");
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
    
    println!("\n\n{}", "=".repeat(50));
    
    // Test 2: Multiple tool calls
    println!("\nTest 2: Multiple tool calls");
    println!("---------------------------");
    let task2 = Task::new(
        "Get the weather in Paris and search for information about the Eiffel Tower. Use both get_weather and web_search tools.".to_string(),
        None,
    );
    
    let handler2 = BasicStreamingHandler;
    let mut stream2 = agent.call_stream_with_handler(task2, handler2).await;
    
    println!("Streaming response:");
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
    
    println!("\n\n✅ All tests completed!");
    
    Ok(())
}
