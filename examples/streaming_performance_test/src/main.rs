use merco_agents::{
    Agent, AgentModelConfig, LlmConfig, Provider, StreamingHandler, StreamingChunk, StreamingResponse,
    Task, AgentRole, AgentCapabilities,
};
use futures::StreamExt;
use std::io::Write;
use dotenv::dotenv;
use colored::*;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use merco_llmproxy::{merco_tool, get_all_tools};
use serde_json::json;

// Performance monitoring streaming handler
struct PerformanceHandler {
    start_time: Instant,
    chunk_times: Arc<Mutex<Vec<Instant>>>,
    tool_times: Arc<Mutex<Vec<(String, Instant, Instant)>>>,
    total_chunks: Arc<Mutex<usize>>,
    total_tools: Arc<Mutex<usize>>,
}

impl PerformanceHandler {
    fn new() -> Self {
        Self {
            start_time: Instant::now(),
            chunk_times: Arc::new(Mutex::new(Vec::new())),
            tool_times: Arc::new(Mutex::new(Vec::new())),
            total_chunks: Arc::new(Mutex::new(0)),
            total_tools: Arc::new(Mutex::new(0)),
        }
    }
    
    fn get_performance_stats(&self) -> (f64, f64, f64, usize, usize) {
        let total_time = self.start_time.elapsed().as_secs_f64();
        let chunk_times = self.chunk_times.lock().unwrap();
        let tool_times = self.tool_times.lock().unwrap();
        let total_chunks = *self.total_chunks.lock().unwrap();
        let total_tools = *self.total_tools.lock().unwrap();
        
        let avg_chunk_time = if !chunk_times.is_empty() {
            chunk_times.iter().map(|t| t.elapsed().as_secs_f64()).sum::<f64>() / chunk_times.len() as f64
        } else {
            0.0
        };
        
        let avg_tool_time = if !tool_times.is_empty() {
            tool_times.iter().map(|(_, start, end)| end.duration_since(*start).as_secs_f64()).sum::<f64>() / tool_times.len() as f64
        } else {
            0.0
        };
        
        (total_time, avg_chunk_time, avg_tool_time, total_chunks, total_tools)
    }
}

impl StreamingHandler for PerformanceHandler {
    fn handle_chunk(&self, chunk: StreamingChunk) {
        let chunk_time = Instant::now();
        *self.total_chunks.lock().unwrap() += 1;
        self.chunk_times.lock().unwrap().push(chunk_time);
        
        if !chunk.content.is_empty() {
            print!("{}", chunk.content);
            std::io::stdout().flush().unwrap();
        }
        
        if chunk.has_tool_calls {
            println!("\n{}", "🔧 Tool calls detected - monitoring performance...".bright_cyan());
        }
    }
    
    fn handle_tool_calls(&self, tool_calls: Vec<merco_agents::agent::ToolCall>) {
        *self.total_tools.lock().unwrap() += tool_calls.len();
        
        println!("\n{}", "🛠️ Tool Performance Analysis:".bright_magenta());
        for (i, call) in tool_calls.iter().enumerate() {
            let tool_start = Instant::now();
            let tool_end = tool_start + std::time::Duration::from_millis(call.execution_time_ms);
            
            self.tool_times.lock().unwrap().push((
                call.tool_name.clone(),
                tool_start,
                tool_end,
            ));
            
            let performance_color = if call.execution_time_ms < 100 {
                "fast".bright_green()
            } else if call.execution_time_ms < 500 {
                "moderate".bright_yellow()
            } else {
                "slow".bright_red()
            };
            
            println!("  {}. {} - {} ({}ms)", 
                (i + 1).to_string().bright_blue(), 
                call.tool_name.bright_cyan(), 
                performance_color,
                call.execution_time_ms.to_string().bright_white()
            );
        }
    }
    
    fn handle_final(&self, response: StreamingResponse) {
        let (total_time, avg_chunk_time, avg_tool_time, total_chunks, total_tools) = self.get_performance_stats();
        
        println!("\n{}", "=".repeat(70).bright_blue());
        println!("{}", "📊 Performance Test Results".bright_green());
        println!("{}", "=".repeat(70).bright_blue());
        
        println!("{} {:.2}s", "Total execution time:".bright_white(), total_time);
        println!("{} {}", "Total chunks processed:".bright_white(), total_chunks.to_string().bright_cyan());
        println!("{} {}", "Total tools executed:".bright_white(), total_tools.to_string().bright_cyan());
        println!("{} {:.3}s", "Average chunk processing time:".bright_white(), avg_chunk_time);
        println!("{} {:.3}s", "Average tool execution time:".bright_white(), avg_tool_time);
        
        // Calculate throughput
        let chunks_per_second = if total_time > 0.0 { total_chunks as f64 / total_time } else { 0.0 };
        let tools_per_second = if total_time > 0.0 { total_tools as f64 / total_time } else { 0.0 };
        
        println!("{} {:.2} chunks/s", "Chunk throughput:".bright_white(), chunks_per_second);
        println!("{} {:.2} tools/s", "Tool throughput:".bright_white(), tools_per_second);
        
        // Performance rating
        let performance_rating = if total_time < 5.0 && avg_tool_time < 0.5 {
            "Excellent".bright_green()
        } else if total_time < 10.0 && avg_tool_time < 1.0 {
            "Good".bright_yellow()
        } else {
            "Needs Optimization".bright_red()
        };
        
        println!("{} {}", "Performance Rating:".bright_white(), performance_rating);
        
        println!("{}", "=".repeat(70).bright_blue());
    }
    
    fn handle_error(&self, error: String) {
        eprintln!("{} {}", "❌ Performance Test Error:".bright_red(), error.bright_red());
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
        2000
    );

    // Get available tools
    let tools = get_all_tools();
    
    // Create agent
    let mut agent = Agent::new(
        "Performance Test Agent".to_string(),
        "An agent for testing streaming performance".to_string(),
        AgentRole::new("Assistant".to_string(), "A helpful AI assistant for performance testing".to_string()),
        config,
        tools, // Now with tools!
        AgentCapabilities {
            max_concurrent_tasks: 5,
            supported_output_formats: vec![merco_agents::agent::role::OutputFormat::Text],
        },
    );
    
    println!("{}", "🚀 Streaming Performance Test".bright_green());
    println!("{}", "=".repeat(50).bright_blue());
    
    // Test 1: Single tool performance
    println!("\n{}", "Test 1: Single Tool Performance".bright_yellow());
    println!("{}", "-".repeat(40));
    let task1 = Task::new(
        "Get the weather in Tokyo and tell me about it.".to_string(),
        None,
    );
    
    let handler1 = PerformanceHandler::new();
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
    
    // Test 2: Multiple tools performance
    println!("\n{}", "Test 2: Multiple Tools Performance".bright_yellow());
    println!("{}", "-".repeat(40));
    let task2 = Task::new(
        "Get weather for 5 different cities: New York, London, Paris, Tokyo, and Sydney. Then search for information about each city's main attractions. Finally, calculate the total cost for visiting all these cities.".to_string(),
        None,
    );
    
    let handler2 = PerformanceHandler::new();
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
    
    // Test 3: Stress test with complex request
    println!("\n{}", "Test 3: Stress Test - Complex Multi-Step Request".bright_yellow());
    println!("{}", "-".repeat(40));
    let task3 = Task::new(
        "I'm planning a world tour. Please get weather for 10 major cities, search for attractions in each, calculate costs, and provide a comprehensive travel plan. Use all available tools multiple times.".to_string(),
        None,
    );
    
    let handler3 = PerformanceHandler::new();
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
    
    println!("\n{}", "✅ Performance tests completed!".bright_green());
    println!("{}", "Check the performance metrics above to evaluate streaming efficiency.".bright_white());
    
    Ok(())
}
