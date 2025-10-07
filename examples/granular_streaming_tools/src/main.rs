use merco_agents::{
    Agent, AgentModelConfig, LlmConfig, StreamingHandler, StreamingChunk, StreamingResponse,
    Task, AgentRole, AgentCapabilities,
};
use futures::StreamExt;
use std::io::Write;
use dotenv::dotenv;
use colored::*;
use merco_llmproxy::{merco_tool, get_all_tools};
use std::collections::HashMap;

// Enhanced streaming handler that demonstrates granular tool call tracking
struct GranularStreamingHandler {
    tool_call_states: std::sync::Arc<std::sync::Mutex<HashMap<String, ToolCallState>>>,
    chunk_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

#[derive(Debug, Clone)]
enum ToolCallState {
    Starting,
    Streaming,
    Ready,
    Executed,
}

impl GranularStreamingHandler {
    fn new() -> Self {
        Self {
            tool_call_states: std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())),
            chunk_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }
}

impl StreamingHandler for GranularStreamingHandler {
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
    }
    
    fn handle_tool_call_start(&self, tool_name: String, call_id: String) {
        println!("\n{}", "🚀 TOOL CALL STARTED".bright_green());
        println!("   {} {}", "Tool:".bright_white(), tool_name.bright_cyan());
        println!("   {} {}", "ID:".bright_white(), call_id.bright_blue());
        println!("   {} {}", "Status:".bright_white(), "Starting...".bright_yellow());
        
        // Track the state
        if let Ok(mut states) = self.tool_call_states.lock() {
            states.insert(call_id.clone(), ToolCallState::Starting);
        }
    }
    
    fn handle_tool_call_streaming(&self, tool_name: String, call_id: String, partial_args: String) {
        println!("\n{}", "📡 TOOL CALL STREAMING".bright_blue());
        println!("   {} {}", "Tool:".bright_white(), tool_name.bright_cyan());
        println!("   {} {}", "ID:".bright_white(), call_id.bright_blue());
        println!("   {} {}", "Partial Args:".bright_white(), partial_args.bright_magenta());
        
        // Update state
        if let Ok(mut states) = self.tool_call_states.lock() {
            states.insert(call_id.clone(), ToolCallState::Streaming);
        }
    }
    
    fn handle_tool_call_ready(&self, tool_name: String, call_id: String, complete_args: String) {
        println!("\n{}", "✅ TOOL CALL READY".bright_green());
        println!("   {} {}", "Tool:".bright_white(), tool_name.bright_cyan());
        println!("   {} {}", "ID:".bright_white(), call_id.bright_blue());
        println!("   {} {}", "Complete Args:".bright_white(), complete_args.bright_green());
        println!("   {} {}", "Status:".bright_white(), "Ready to execute!".bright_green());
        
        // Update state
        if let Ok(mut states) = self.tool_call_states.lock() {
            states.insert(call_id.clone(), ToolCallState::Ready);
        }
    }
    
    fn handle_tool_call_executed(&self, tool_name: String, call_id: String, result: String, execution_time_ms: u64) {
        println!("\n{}", "🎯 TOOL CALL EXECUTED".bright_magenta());
        println!("   {} {}", "Tool:".bright_white(), tool_name.bright_cyan());
        println!("   {} {}", "ID:".bright_white(), call_id.bright_blue());
        println!("   {} {}", "Result:".bright_white(), result.bright_white());
        println!("   {} {}ms", "Execution Time:".bright_white(), execution_time_ms.to_string().bright_yellow());
        println!("   {} {}", "Status:".bright_white(), "Completed!".bright_green());
        
        // Update state
        if let Ok(mut states) = self.tool_call_states.lock() {
            states.insert(call_id.clone(), ToolCallState::Executed);
        }
    }
    
    fn handle_tool_calls(&self, tool_calls: Vec<merco_agents::agent::ToolCall>) {
        println!("\n{}", "🛠️ FINAL TOOL CALLS SUMMARY".bright_magenta());
        for (i, call) in tool_calls.iter().enumerate() {
            println!("  {}. {} - {}", 
                (i + 1).to_string().bright_blue(), 
                call.tool_name.bright_cyan(), 
                call.parameters.bright_white()
            );
            println!("     {} {}", "Result:".bright_green(), call.result.bright_white());
            println!("     {} {}ms", "Execution time:".dimmed(), call.execution_time_ms.to_string().bright_yellow());
        }
    }
    
    fn handle_final(&self, response: StreamingResponse) {
        println!("\n{}", "=".repeat(60).bright_blue());
        println!("{}", "📊 GRANULAR STREAMING COMPLETE".bright_green());
        println!("{}", "=".repeat(60).bright_blue());
        println!("{} {}", "Total tokens:".bright_white(), response.total_tokens.to_string().bright_yellow());
        println!("{} {}ms", "Execution time:".bright_white(), response.execution_time_ms.to_string().bright_yellow());
        println!("{} {}", "Model:".bright_white(), response.model_used.bright_cyan());
        println!("{} {}", "Success:".bright_white(), if response.success { "✅".bright_green() } else { "❌".bright_red() });
        
        if !response.tools_used.is_empty() {
            println!("{} {:?}", "Tools used:".bright_white(), response.tools_used.join(", ").bright_cyan());
        }
        
        // Show final tool call states
        if let Ok(states) = self.tool_call_states.lock() {
            println!("\n{}", "🔍 TOOL CALL STATES:".bright_white());
            for (call_id, state) in states.iter() {
                let state_str = match state {
                    ToolCallState::Starting => "Starting".bright_yellow(),
                    ToolCallState::Streaming => "Streaming".bright_blue(),
                    ToolCallState::Ready => "Ready".bright_green(),
                    ToolCallState::Executed => "Executed".bright_magenta(),
                };
                println!("   {} {}", call_id.bright_blue(), state_str);
            }
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
        merco_agents::Provider::OpenAI,
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
        "Granular Streaming Agent".to_string(),
        "An agent that demonstrates granular tool call streaming".to_string(),
        AgentRole::new("Assistant".to_string(), "A helpful AI assistant with detailed streaming capabilities".to_string()),
        config,
        tools,
        AgentCapabilities {
            max_concurrent_tasks: 3,
            supported_output_formats: vec![merco_agents::agent::role::OutputFormat::Text],
        },
    );
    
    println!("{}", "🚀 Granular Tool Call Streaming Example".bright_green());
    println!("{}", "=".repeat(50).bright_blue());
    println!("{}", "This example shows the different phases of tool calls:".bright_white());
    println!("   {} {}", "🚀".bright_green(), "Tool call starts (tool name known)".bright_white());
    println!("   {} {}", "📡".bright_blue(), "Parameters streaming (LLM filling args)".bright_white());
    println!("   {} {}", "✅".bright_green(), "Tool call ready (args complete)".bright_white());
    println!("   {} {}", "🎯".bright_magenta(), "Tool call executed (result available)".bright_white());
    println!("{}", "=".repeat(50).bright_blue());
    
    // Test with a complex scenario
    println!("\n{}", "Test: Complex Multi-Tool Scenario".bright_yellow());
    println!("{}", "-".repeat(40));
    let task = Task::new(
        "I need to plan a trip to Japan. Please get the weather in Tokyo, search for information about cherry blossoms, and calculate the cost for a 7-day trip. Use all available tools to help me plan.".to_string(),
        None,
    );
    
    let handler = GranularStreamingHandler::new();
    let mut stream = agent.call_stream_with_handler(task, handler).await;
    
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
    
    println!("\n{}", "✅ Granular streaming test completed!".bright_green());
    
    Ok(())
}
