use merco_agents::agent::agent::{Agent, AgentLLMConfig};
use merco_agents::task::task::Task;
use merco_llmproxy::LlmConfig;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables
    dotenv::dotenv().ok();
    
    // Get API key from environment
    let api_key = env::var("OPENROUTER_API_KEY")
        .expect("Please set OPENROUTER_API_KEY environment variable");
    
    println!("🤖 Basic Agent Example");
    println!("=====================");
    
    // Create LLM configuration
    let llm_config = LlmConfig {
        provider: merco_llmproxy::config::Provider::OpenAI,
        api_key: Some(api_key),
        base_url: Some("https://openrouter.ai/api/v1".to_string()),
    };
    
    let agent_llm_config = AgentLLMConfig::new(
        llm_config,
        "openai/gpt-4o-mini".to_string(),
        0.7,
        1000,
    );
    
    // Create a basic agent
    let mut agent = Agent::new(
        agent_llm_config,
        "You are a helpful AI assistant that provides clear and concise answers.".to_string(),
        vec![
            "Help users with their questions".to_string(),
            "Provide accurate information".to_string(),
            "Be friendly and professional".to_string(),
        ],
        vec![], // No tools for this basic example
    );
    
    println!("✅ Agent created successfully!");
    println!("Agent ID: {}", agent.get_id());
    println!("Agent Name: {}", agent.get_name());
    println!("Agent Status: {:?}", agent.get_state().status);
    
    // Create a simple task
    let task = Task::new(
        "Explain what artificial intelligence is in simple terms".to_string(),
        Some("A clear, beginner-friendly explanation of AI".to_string()),
    );
    
    println!("\n📝 Executing task...");
    println!("Task: {}", task.description);
    
    // Execute the task (now returns AgentResponse by default)
    println!("\n📝 Executing task...");
    let response = agent.call(task).await;
    
    if response.success {
        println!("✅ Task completed successfully!");
        println!("Response: {}", response.content);
        println!("\n📊 Execution Metrics:");
        println!("  - Execution time: {}ms", response.execution_time_ms);
        println!("  - Input tokens: {}", response.input_tokens);
        println!("  - Output tokens: {}", response.output_tokens);
        println!("  - Total tokens: {}", response.total_tokens);
        println!("  - Tokens per second: {:.2}", response.tokens_per_second());
        println!("  - Model used: {}", response.model_used);
        println!("  - Temperature: {}", response.temperature);
        if !response.tools_used.is_empty() {
            println!("  - Tools used: {:?}", response.tools_used);
        }
    } else {
        println!("❌ Task failed: {}", response.error.unwrap_or("Unknown error".to_string()));
    }
    
    // Test string input method (now returns AgentResponse by default)
    println!("\n🔤 Testing string input method...");
    let str_response = agent.call_str("What are the benefits of using AI in everyday life?").await;
    
    if str_response.success {
        println!("✅ String input successful!");
        println!("Response: {}", str_response.content);
        println!("\n📊 String Input Metrics:");
        println!("  - Execution time: {}ms", str_response.execution_time_ms);
        println!("  - Total tokens: {}", str_response.total_tokens);
        println!("  - Tokens per second: {:.2}", str_response.tokens_per_second());
    } else {
        println!("❌ String input failed: {}", str_response.error.unwrap_or("Unknown error".to_string()));
    }
    
    // Show agent performance metrics
    println!("\n📊 Agent Performance Summary:");
    let metrics = agent.get_performance_metrics();
    println!("Total tasks: {}", metrics.total_tasks);
    println!("Successful tasks: {}", metrics.successful_tasks);
    println!("Failed tasks: {}", metrics.failed_tasks);
    println!("Success rate: {:.2}%", agent.get_success_rate() * 100.0);
    
    println!("\n🎉 Basic agent example completed!");
    Ok(())
}
