use merco_agents::agent::agent::{Agent, AgentLLMConfig};
use merco_agents::agent::role::{OutputFormat, AgentRole, AgentCapabilities, ProcessingMode};
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
    
    println!("👥 Multi-Agent Example - Independent Agents");
    println!("===========================================");
    
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
    
    // Create specialized independent agents
    let mut agents = vec![
        // Research Agent
        {
            let role = AgentRole::new(
                "Research Specialist".to_string(),
                "You are an expert researcher who gathers and analyzes information from various sources.".to_string(),
            );
            let capabilities = AgentCapabilities {
                max_concurrent_tasks: 1,
                supported_output_formats: vec![OutputFormat::Text, OutputFormat::Json],
                processing_modes: vec![ProcessingMode::Sequential],
            };
            Agent::with_custom_role(
                "Research Agent".to_string(),
                "Specializes in gathering and analyzing information".to_string(),
                role,
                agent_llm_config.clone(),
                vec![], // No tools for simplicity
                capabilities,
                Some(OutputFormat::Text),
            )
        },
        // Analysis Agent
        {
            let role = AgentRole::new(
                "Data Analyst".to_string(),
                "You are a data analyst who processes information and provides insights.".to_string(),
            );
            let capabilities = AgentCapabilities {
                max_concurrent_tasks: 1,
                supported_output_formats: vec![OutputFormat::Json, OutputFormat::Markdown],
                processing_modes: vec![ProcessingMode::Sequential],
            };
            Agent::with_custom_role(
                "Analysis Agent".to_string(),
                "Specializes in data analysis and insights".to_string(),
                role,
                agent_llm_config.clone(),
                vec![],
                capabilities,
                Some(OutputFormat::Json),
            )
        },
        // Writing Agent
        {
            let role = AgentRole::new(
                "Content Writer".to_string(),
                "You are a professional writer who creates engaging and well-structured content.".to_string(),
            );
            let capabilities = AgentCapabilities {
                max_concurrent_tasks: 1,
                supported_output_formats: vec![OutputFormat::Markdown, OutputFormat::Html],
                processing_modes: vec![ProcessingMode::Sequential],
            };
            Agent::with_custom_role(
                "Writing Agent".to_string(),
                "Specializes in content creation and writing".to_string(),
                role,
                agent_llm_config.clone(),
                vec![],
                capabilities,
                Some(OutputFormat::Markdown),
            )
        },
    ];
    
    println!("✅ Created {} independent specialized agents", agents.len());
    
    // Example 1: Independent Task Execution
    println!("\n📋 Example 1: Independent Task Execution");
    println!("{}", "=".repeat(50));
    
    let independent_tasks = vec![
        ("Research Agent", "Research the latest trends in artificial intelligence"),
        ("Analysis Agent", "Analyze the economic impact of remote work"),
        ("Writing Agent", "Write a short article about sustainable living"),
    ];
    
    for (agent_name, task_description) in independent_tasks {
        println!("\n🔍 {} working on: {}", agent_name, task_description);
        
        let task = Task::new(task_description.to_string(), None);
        let agent_index = match agent_name {
            "Research Agent" => 0,
            "Analysis Agent" => 1,
            "Writing Agent" => 2,
            _ => 0,
        };
        
        let response = agents[agent_index].call(task).await;
        
        if response.success {
            println!("✅ {} completed successfully!", agent_name);
            println!("📊 Metrics: {}ms, {} tokens, {} tokens/sec", 
                response.execution_time_ms, 
                response.total_tokens,
                response.tokens_per_second()
            );
            println!("📝 Output: {}", response.content);
        } else {
            println!("❌ {} failed: {}", agent_name, response.error.unwrap_or("Unknown error".to_string()));
        }
    }
    
    // Example 2: Simple String Input
    println!("\n⚡ Example 2: Simple String Input");
    println!("{}", "=".repeat(50));
    
    let string_inputs = vec![
        ("Research Agent", "What are the main types of renewable energy?"),
        ("Analysis Agent", "What are the environmental benefits of solar power?"),
        ("Writing Agent", "What are the economic benefits of wind energy?"),
    ];
    
    for (agent_name, input) in string_inputs {
        println!("\n🔍 {} working on: {}", agent_name, input);
        
        let agent_index = match agent_name {
            "Research Agent" => 0,
            "Analysis Agent" => 1,
            "Writing Agent" => 2,
            _ => 0,
        };
        
        let response = agents[agent_index].call_str(input).await;
        
        if response.success {
            println!("✅ {} completed successfully!", agent_name);
            println!("📊 Metrics: {}ms, {} tokens", 
                response.execution_time_ms, response.total_tokens);
            println!("📝 Output: {}", response.content);
        } else {
            println!("❌ {} failed: {}", agent_name, response.error.unwrap_or("Unknown error".to_string()));
        }
    }
    
    // Example 3: Simple Workflow
    println!("\n🔄 Example 3: Simple Workflow");
    println!("{}", "=".repeat(50));
    
    // Each agent works on their own specialized task independently
    let workflow_tasks = vec![
        ("Research Agent", "Research the benefits of electric vehicles"),
        ("Analysis Agent", "Analyze the data from the research phase"),
        ("Writing Agent", "Create a summary report based on the analysis"),
    ];
    
    let mut workflow_results = Vec::new();
    
    for (agent_name, task_description) in workflow_tasks {
        println!("\n🔍 {} working on: {}", agent_name, task_description);
        
        let task = Task::new(task_description.to_string(), None);
        let agent_index = match agent_name {
            "Research Agent" => 0,
            "Analysis Agent" => 1,
            "Writing Agent" => 2,
            _ => 0,
        };
        
        let response = agents[agent_index].call(task).await;
        
        if response.success {
            println!("✅ {} completed successfully!", agent_name);
            println!("📊 Metrics: {}ms, {} tokens", 
                response.execution_time_ms, response.total_tokens);
            workflow_results.push(response.content);
        } else {
            println!("❌ {} failed: {}", agent_name, response.error.unwrap_or("Unknown error".to_string()));
            workflow_results.push("Failed".to_string());
        }
    }
    
    // Show individual agent performance
    println!("\n📊 Individual Agent Performance:");
    println!("{}", "=".repeat(40));
    
    for (i, agent) in agents.iter().enumerate() {
        let metrics = agent.get_performance_metrics();
        println!("Agent {} ({}):", i + 1, agent.get_name());
        println!("  - Total tasks: {}", metrics.total_tasks);
        println!("  - Successful: {}", metrics.successful_tasks);
        println!("  - Failed: {}", metrics.failed_tasks);
        println!("  - Success rate: {:.2}%", agent.get_success_rate() * 100.0);
        println!("  - Average response time: {:.2}ms", metrics.average_response_time_ms);
        println!("  - Total tokens used: {}", metrics.average_tokens_used);
    }
    
    // Show workflow results
    println!("\n📋 Workflow Results Summary:");
    println!("{}", "=".repeat(40));
    for (i, result) in workflow_results.iter().enumerate() {
        println!("Step {}: {}", i + 1, result);
    }
    
    println!("\n🎉 Multi-agent example completed!");
    println!("All agents worked independently and efficiently!");
    
    Ok(())
}