use merco_agents::{Agent, AgentModelConfig, OutputFormat, AgentRole, AgentCapabilities, ProcessingMode, Task, Provider, LlmConfig};
use merco_agents::task::task::{JsonField, JsonFieldType};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables
    dotenv::dotenv().ok();
    
    // Get API key from environment
    let api_key = env::var("OPENROUTER_API_KEY")
        .expect("Please set OPENROUTER_API_KEY environment variable");
    
    println!("🎨 Output Format Example");
    println!("=======================");
    
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
    
    // Test different output formats
    let formats = vec![
        (OutputFormat::Text, "Plain Text"),
        (OutputFormat::Json, "JSON"),
        (OutputFormat::Markdown, "Markdown"),
        (OutputFormat::Html, "HTML"),
    ];
    
    for (format, format_name) in formats {
        println!("\n🔧 Testing {} Format", format_name);
        println!("{}", "=".repeat(30));
        
        // Create agent with specific output format
        let role = AgentRole::new(
            "Data Analyst".to_string(),
            "You are a data analyst that presents information clearly.".to_string(),
        );
        let capabilities = AgentCapabilities {
            max_concurrent_tasks: 1,
            supported_output_formats: vec![format.clone()],
            processing_modes: vec![ProcessingMode::Sequential],
        };
        
        let mut agent = Agent::new_with_output_format(
            "Data Analyst".to_string(),
            "Specializes in data analysis and insights".to_string(),
            role,
            agent_llm_config.clone(),
            vec![], // No tools
            capabilities,
            format.clone(),
        );
        
        // Create a task that matches the agent's format
        let task = match format {
            OutputFormat::Json => {
                let required_fields = vec![
                    JsonField {
                        name: "summary".to_string(),
                        field_type: JsonFieldType::String,
                        description: Some("Brief summary of the topic".to_string()),
                    },
                    JsonField {
                        name: "benefits".to_string(),
                        field_type: JsonFieldType::Array(Box::new(JsonFieldType::String)),
                        description: Some("List of key benefits".to_string()),
                    },
                ];
                Task::new_with_json_output(
                    "Create a summary of the benefits of renewable energy".to_string(),
                    Some("A JSON object with key benefits and statistics".to_string()),
                    required_fields,
                    vec![],
                    false,
                )
            },
            OutputFormat::Markdown => Task::new(
                "Write a guide about sustainable living practices".to_string(),
                Some("A well-formatted markdown guide with headers and lists".to_string()),
            ),
            OutputFormat::Html => Task::new(
                "Create a product comparison table for electric vehicles".to_string(),
                Some("An HTML table with proper styling".to_string()),
            ),
            _ => Task::new(
                "Explain the concept of machine learning".to_string(),
                Some("A clear explanation suitable for beginners".to_string()),
            ),
        };
        
        println!("Task: {}", task.description);
        println!("Expected Format: {:?}", task.output_format);
        
        // Execute the task (now returns AgentResponse by default)
        let response = agent.call(task).await;
        
        if response.success {
            println!("✅ Task completed successfully!");
            println!("Response ({}):", format_name);
            println!("{}", response.content);
            println!("📊 Metrics: {}ms, {} tokens, {:.2} tokens/sec", 
                response.execution_time_ms, 
                response.total_tokens, 
                response.tokens_per_second());
        } else {
            println!("❌ Task failed: {}", response.error.unwrap_or("Unknown error".to_string()));
        }
        
        // Show agent's configured format
        println!("Agent's default format: {:?}", agent.output_handler.default_format);
    }
    
    // Test format override (agent with different format than task)
    println!("\n🔄 Testing Format Override");
    println!("{}", "=".repeat(30));
    
    // Create agent with JSON format
    let role = AgentRole::new(
        "JSON Specialist".to_string(),
        "You are a JSON specialist.".to_string(),
    );
    let capabilities = AgentCapabilities {
        max_concurrent_tasks: 1,
        supported_output_formats: vec![OutputFormat::Json, OutputFormat::Markdown],
        processing_modes: vec![ProcessingMode::Sequential],
    };
    
    let mut json_agent = Agent::new_with_output_format(
        "JSON Specialist".to_string(),
        "Specializes in JSON data formatting".to_string(),
        role,
        agent_llm_config.clone(),
        vec![],
        capabilities,
        OutputFormat::Json,
    );
    
    // Create task with Markdown format (different from agent)
    let markdown_task = Task::new(
        "Write a recipe for chocolate cake".to_string(),
        Some("A well-formatted markdown recipe".to_string()),
    );
    
    println!("Agent format: {:?}", json_agent.output_handler.default_format);
    println!("Task format: {:?}", markdown_task.output_format);
    println!("Expected: Task format should override agent format");
    
    let response = json_agent.call(markdown_task).await;
    
    if response.success {
        println!("✅ Format override successful!");
        println!("Response:");
        println!("{}", response.content);
        println!("📊 Metrics: {}ms, {} tokens, {:.2} tokens/sec", 
            response.execution_time_ms, 
            response.total_tokens, 
            response.tokens_per_second());
    } else {
        println!("❌ Format override failed: {}", response.error.unwrap_or("Unknown error".to_string()));
    }
    
    println!("\n🎉 Output format example completed!");
    Ok(())
}