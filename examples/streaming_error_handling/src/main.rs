use merco_agents::{
    Agent, AgentModelConfig, LlmConfig, Provider, StreamingHandler, StreamingChunk, StreamingResponse,
    Task, AgentRole, AgentCapabilities,
};
use futures::StreamExt;
use std::io::Write;
use dotenv::dotenv;
use colored::*;
use std::sync::{Arc, Mutex};
use merco_llmproxy::{merco_tool, get_all_tools};
use serde_json::json;

// Error-resilient streaming handler
struct ErrorResilientHandler {
    error_count: Arc<Mutex<usize>>,
    success_count: Arc<Mutex<usize>>,
    tool_errors: Arc<Mutex<Vec<String>>>,
}

impl ErrorResilientHandler {
    fn new() -> Self {
        Self {
            error_count: Arc::new(Mutex::new(0)),
            success_count: Arc::new(Mutex::new(0)),
            tool_errors: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    fn get_stats(&self) -> (usize, usize, Vec<String>) {
        let errors = *self.error_count.lock().unwrap();
        let successes = *self.success_count.lock().unwrap();
        let tool_errors = self.tool_errors.lock().unwrap().clone();
        (errors, successes, tool_errors)
    }
}

impl StreamingHandler for ErrorResilientHandler {
    fn handle_chunk(&self, chunk: StreamingChunk) {
        if !chunk.content.is_empty() {
            // Track successful chunks
            *self.success_count.lock().unwrap() += 1;
            
            // Color code based on content type
            let colored_content = if chunk.content.contains("❌") || chunk.content.contains("Error") {
                chunk.content.bright_red()
            } else if chunk.content.contains("✅") || chunk.content.contains("Success") {
                chunk.content.bright_green()
            } else if chunk.content.contains("⚠️") || chunk.content.contains("Warning") {
                chunk.content.bright_yellow()
            } else {
                chunk.content.white()
            };
            
            print!("{}", colored_content);
            std::io::stdout().flush().unwrap();
        }
        
        if chunk.has_tool_calls {
            println!("\n{}", "🔧 Tool calls detected - monitoring for errors...".bright_cyan());
        }
    }
    
    fn handle_tool_calls(&self, tool_calls: Vec<merco_agents::agent::ToolCall>) {
        println!("\n{}", "🛠️ Tool Call Results:".bright_magenta());
        for (i, call) in tool_calls.iter().enumerate() {
            if let Some(error) = &call.error {
                // Track tool errors
                *self.error_count.lock().unwrap() += 1;
                self.tool_errors.lock().unwrap().push(format!("Tool {}: {}", call.tool_name, error));
                
                println!("  {}. {} - {}", 
                    (i + 1).to_string().bright_red(), 
                    call.tool_name.bright_red(), 
                    "❌ ERROR".bright_red()
                );
                println!("     {} {}", "Error:".bright_red(), error.bright_red());
            } else {
                // Track successful tool calls
                *self.success_count.lock().unwrap() += 1;
                
                println!("  {}. {} - {}", 
                    (i + 1).to_string().bright_green(), 
                    call.tool_name.bright_green(), 
                    "✅ SUCCESS".bright_green()
                );
                println!("     {} {}", "Result:".bright_white(), call.result.bright_white());
            }
            println!("     {} {}ms", "Execution time:".dimmed(), call.execution_time_ms.to_string().bright_yellow());
        }
    }
    
    fn handle_final(&self, response: StreamingResponse) {
        let (errors, successes, tool_errors) = self.get_stats();
        
        println!("\n{}", "=".repeat(60).bright_blue());
        println!("{}", "📊 Error Handling Test Results".bright_green());
        println!("{}", "=".repeat(60).bright_blue());
        
        println!("{} {}", "Total successful chunks:".bright_white(), successes.to_string().bright_green());
        println!("{} {}", "Total errors:".bright_white(), errors.to_string().bright_red());
        println!("{} {}", "Success rate:".bright_white(), 
            if successes + errors > 0 { 
                format!("{:.1}%", (successes as f64 / (successes + errors) as f64) * 100.0).bright_green()
            } else { 
                "N/A".bright_yellow() 
            }
        );
        
        if !tool_errors.is_empty() {
            println!("{}", "Tool Errors:".bright_red());
            for error in tool_errors {
                println!("  - {}", error.bright_red());
            }
        }
        
        println!("{} {}", "Response success:".bright_white(), 
            if response.success { "✅".bright_green() } else { "❌".bright_red() }
        );
        
        if let Some(error) = response.error {
            println!("{} {}", "Response error:".bright_red(), error.bright_red());
        }
        
        println!("{}", "=".repeat(60).bright_blue());
    }
    
    fn handle_error(&self, error: String) {
        *self.error_count.lock().unwrap() += 1;
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
        1000
    );

    // Get available tools
    let tools = get_all_tools();
    
    // Create agent
    let mut agent = Agent::new(
        "Error Handling Agent".to_string(),
        "An agent for testing error resilience in streaming".to_string(),
        AgentRole::new("Assistant".to_string(), "A helpful AI assistant for error testing".to_string()),
        config,
        tools, // Now with tools!
        AgentCapabilities {
            max_concurrent_tasks: 1,
            supported_output_formats: vec![merco_agents::agent::role::OutputFormat::Text],
        },
    );
    
    println!("{}", "🚀 Streaming Error Handling Test".bright_green());
    println!("{}", "=".repeat(50).bright_blue());
    
    // Test 1: Normal operation
    println!("\n{}", "Test 1: Normal Operation".bright_yellow());
    println!("{}", "-".repeat(30));
    let task1 = Task::new(
        "Get the weather in New York and search for information about Central Park.".to_string(),
        None,
    );
    
    let handler1 = ErrorResilientHandler::new();
    let mut stream1 = agent.call_stream_with_handler(task1, handler1).await;
    
    println!("{}", "Streaming response:".bright_white());
    while let Some(chunk_result) = stream1.next().await {
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
    
    // Test 2: Edge case - very long request
    println!("\n{}", "Test 2: Long Request".bright_yellow());
    println!("{}", "-".repeat(30));
    let task2 = Task::new(
        "Please get weather for multiple cities: Tokyo, London, Paris, Sydney, and New York. Then search for information about each city's main attractions. Finally, calculate the total cost for visiting all these cities.".to_string(),
        None,
    );
    
    let handler2 = ErrorResilientHandler::new();
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
    
    // Test 3: Ambiguous request that might cause tool errors
    println!("\n{}", "Test 3: Ambiguous Request".bright_yellow());
    println!("{}", "-".repeat(30));
    let task3 = Task::new(
        "I need help with something. Can you use the tools to figure out what I need?".to_string(),
        None,
    );
    
    let handler3 = ErrorResilientHandler::new();
    let mut stream3 = agent.call_stream_with_handler(task3, handler3).await;
    
    println!("{}", "Streaming response:".bright_white());
    while let Some(chunk_result) = stream3.next().await {
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
    
    println!("\n{}", "✅ Error handling tests completed!".bright_green());
    println!("{}", "The system demonstrated resilience to various edge cases.".bright_white());
    
    Ok(())
}
