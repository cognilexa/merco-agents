use merco_agents::agent::agent::{Agent, AgentLLMConfig};
use merco_agents::agent::role::{OutputFormat, AgentRole, AgentCapabilities, ProcessingMode};
use merco_agents::task::task::Task;
use merco_llmproxy::LlmConfig;
use merco_llmproxy::merco_tool;
use merco_llmproxy::get_all_tools;
use std::env;
use serde_json::json;
use chrono::Utc;

// Define tools using merco_tool macro from merco-llmproxy
#[merco_tool(description = "Perform mathematical calculations")]
fn calculate(operation: String, a: f64, b: f64) -> serde_json::Value {
    let result = match operation.as_str() {
        "add" => a + b,
        "subtract" => a - b,
        "multiply" => a * b,
        "divide" => if b != 0.0 { a / b } else { return json!({"error": "Division by zero"}) },
        _ => return json!({"error": format!("Unknown operation: {}", operation)}),
    };

    println!("!!!!!!!!!!!!!!!! calculate result: {}", result);
    
    json!({
        "result": result,
        "operation": operation,
        "inputs": {"a": a, "b": b}
    })
}

#[merco_tool(description = "Get weather information for a city")]
fn get_weather(city: String) -> serde_json::Value {
    // Simulate weather data
    println!("get_weather result: {}", city);
    json!({
        "city": city,
        "temperature": "22°C",
        "condition": "Sunny",
        "humidity": "65%",
        "wind": "10 km/h"
    })
}

#[merco_tool(description = "Get the current time")]
fn get_current_time() -> serde_json::Value {
    let now = Utc::now();
    println!("get_current_time result: {}", now.to_rfc3339());
    json!({
        "timestamp": now.to_rfc3339(),
        "formatted": now.format("%Y-%m-%d %H:%M:%S UTC").to_string()
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables
    dotenv::dotenv().ok();
    
    // Get API key from environment
    let api_key = env::var("OPENROUTER_API_KEY")
        .expect("Please set OPENROUTER_API_KEY environment variable");
    
    println!("🔧 Tool Usage Example with Merco Macros");
    println!("=======================================");
    
    // Create LLM configuration
    let llm_config = LlmConfig {
        provider: merco_llmproxy::config::Provider::OpenAI,
        base_url: Some("https://openrouter.ai/api/v1".to_string()),
        api_key: Some(api_key),
    };
    
    let agent_llm_config = AgentLLMConfig::new(
        llm_config,
        "openai/gpt-4o-mini".to_string(),
        0.7,
        1000,
    );
    
    // Create agent with tools (tools are automatically registered via macros)
    let role = AgentRole::new(
        "Tool Assistant".to_string(),
        "You are a helpful assistant with access to various tools. Use them when appropriate to help users.".to_string(),
    );
    let capabilities = AgentCapabilities {
        max_concurrent_tasks: 1,
        supported_output_formats: vec![OutputFormat::Text, OutputFormat::Json],
        processing_modes: vec![ProcessingMode::Sequential],
    };
    
    // Get all registered tools from the global registry
    let available_tools = merco_llmproxy::get_all_tools();
    println!("📋 Available tools from registry: {:?}", available_tools.iter().map(|t| &t.name).collect::<Vec<_>>());
    
    let mut agent = Agent::new(
        "Tool Assistant".to_string(),
        "Specializes in using tools to help users".to_string(),
        role,
        agent_llm_config,
        available_tools, // Use tools from the global registry
        capabilities,
    );
    
    println!("✅ Agent created with merco tool macros");
    println!("Available tools: calculate, get_weather, get_current_time");
    
    // Test tasks that require tools
    let tasks = vec![
        "What is 15 multiplied by 23?",
        "What's the weather like in Paris?",
        "What time is it now?",
        "Calculate 100 divided by 4, then tell me the current time",
    ];
    
    for (i, task_description) in tasks.iter().enumerate() {
        println!("\n📝 Task {}: {}", i + 1, task_description);
        println!("{}", "-".repeat(50));
        
        let task = Task::new(
            task_description.to_string(),
            Some("Use appropriate tools to provide accurate information".to_string()),
        );
        
        let response = agent.call(task).await;
        
        if response.success {
            println!("✅ Task completed successfully!");
            println!("Response: {}", response.content);
            println!("📊 Metrics: {}ms, {} tokens, {:.2} tokens/sec", 
                response.execution_time_ms, 
                response.total_tokens, 
                response.tokens_per_second());
            
            // Show detailed tool information
            if !response.tool_calls.is_empty() {
                println!("🔧 Tool Calls ({})", response.tool_calls_count);
                for (i, tool_call) in response.tool_calls.iter().enumerate() {
                    println!("  {}. {} ({}ms)", i + 1, tool_call.tool_name, tool_call.execution_time_ms);
                    println!("     Parameters: {}", tool_call.parameters);
                    println!("     Result: {}", tool_call.result);
                    if let Some(error) = &tool_call.error {
                        println!("     Error: {}", error);
                    }
                }
                println!("  Total tool execution time: {}ms", response.tool_execution_time_ms);
            } else if !response.tools_used.is_empty() {
                println!("🔧 Tools used: {}", response.tools_used.join(", "));
            }
            
            // Show output format
            println!("📋 Output format: {}", response.output_format);
        } else {
            println!("❌ Task failed: {}", response.error.unwrap_or("Unknown error".to_string()));
        }
    }
    
    // Show agent performance
    println!("\n📊 Agent Performance:");
    let metrics = agent.get_performance_metrics();
    println!("Total tasks: {}", metrics.total_tasks);
    println!("Successful tasks: {}", metrics.successful_tasks);
    println!("Failed tasks: {}", metrics.failed_tasks);
    println!("Success rate: {:.2}%", agent.get_success_rate() * 100.0);
    
    println!("\n🎉 Tool usage example with merco macros completed!");
    Ok(())
}