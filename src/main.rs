use merco_agents::agent::agent::{Agent, AgentLLMConfig, AgentMemoryConfig};
use merco_agents::task::task::{Task, JsonFieldType};
use merco_agents::memory::{config::{MemoryConfig, EmbeddingProvider, StorageBackend}, AgentMemory, MemoryType};
use merco_llmproxy::{LlmConfig, Provider, get_tools_by_names, merco_tool};

use dotenv::dotenv;
use std::collections::HashMap;

#[merco_tool(description = "A tool to get the current time")]
pub fn get_current_time() -> String {
    println!("get_current_time tool called");
    
    // Get the current system time
    let now = std::time::SystemTime::now();
    // Convert SystemTime to DateTime<Local> using the chrono crate
    let datetime: chrono::DateTime<chrono::Local> = now.into();
    // Format the datetime into a human-readable string
    datetime.format("%Y-%m-%d %H:%M:%S %Z").to_string()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv().ok();

    let api_key = std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY must be set");

    let llm_config = LlmConfig::new(Provider::OpenAI)
        .with_base_url("https://openrouter.ai/api/v1".to_string())
        .with_api_key(api_key);

    let model: &str = "openai/gpt-4o-mini";
    let agent_llm_config = AgentLLMConfig::new(llm_config, model.to_string(), 0.0, 1000);

    println!("🤖 Merco-Agents: Memory-Enhanced Agent System Demo\n");

    // ================================
    // MEMORY-ENABLED AGENT SHOWCASE
    // ================================
    println!("=== 🧠 Memory-Enhanced Agent Demonstration ===");
    
    // Create agent with memory capabilities
    let memory_config = MemoryConfig {
        embedding: EmbeddingProvider::HuggingFace, // Fast, deterministic embeddings
        storage: StorageBackend::SQLiteInMemory,   // SQLite file storage
        limits: Default::default(),
    };
    
    // Set a custom SQLite path for visibility (use absolute path)
    let current_dir = std::env::current_dir()?;
    let db_path = current_dir.join("agent_memory.db");
    std::env::set_var("SQLITE_PATH", db_path.to_string_lossy().to_string());
    
    // Debug: Show storage configuration
    let storage_config = memory_config.storage_config();
    println!("💾 Storage Configuration:");
    println!("  Metadata type: {}", storage_config.metadata_type);
    println!("  Metadata URL: {}", storage_config.metadata_url);
    println!("  Vector type: {}", storage_config.vector_type);
    println!("  Vector URL: {}", storage_config.vector_url);
    println!();
    
    let agent_memory_config = AgentMemoryConfig::new()
        .with_memory_config(memory_config)
        .with_auto_store(true)   // Automatically store interactions
        .with_auto_retrieve(true) // Automatically retrieve context
        .with_context_limit(3);   // Max 3 memories for context
    
    let mut memory_agent = Agent::with_memory(
        agent_llm_config.clone(),
        "You are an intelligent assistant with memory capabilities. You remember user interactions and learn from them to provide better responses.".to_string(),
        vec![
            "Provide helpful and accurate responses".to_string(),
            "Remember user preferences and context".to_string(),
            "Learn from interactions to improve over time".to_string(),
        ],
        vec![], // No tools for this demo
        agent_memory_config
    ).await.map_err(|e| anyhow::anyhow!("Failed to create agent with memory: {}", e))?;

    println!("Memory stats: {:?}", memory_agent.get_memory_stats());

    // Test 1: First interaction with user introduction
    println!("\n=== Test 1: User Introduction ===");
    let task1 = Task::new(
        "Hello! My name is Samet and I prefer detailed explanations with examples when learning new concepts.".to_string(),
        Some("A friendly greeting response".to_string()),
    );

    let result1 = memory_agent.call_with_user(task1, Some("samet".to_string()))
        .await.map_err(|e| anyhow::anyhow!(e))?;
    println!("Response: {}\n", &result1[..std::cmp::min(result1.len(), 300)]);

    // Test 2: Follow-up question (should remember user's name and preferences)
    println!("=== Test 2: Follow-up Question ===");
    let task2 = Task::new(
        "Can you explain what machine learning is?".to_string(),
        Some("An explanation tailored to the user's preferences".to_string()),
    );

    let result2 = memory_agent.call_with_user(task2, Some("samet".to_string()))
        .await.map_err(|e| anyhow::anyhow!(e))?;
    println!("Contextual response: {}\n", &result2[..std::cmp::min(result2.len(), 400)]);

    // Test 3: Check if agent learned something
    println!("=== Test 3: Teaching the Agent ===");
    let task3 = Task::new(
        "I really like programming in Rust because it's both safe and fast. Please remember this preference.".to_string(),
        Some("Acknowledgment of learning the preference".to_string()),
    );

    let result3 = memory_agent.call_with_user(task3, Some("samet".to_string()))
        .await.map_err(|e| anyhow::anyhow!(e))?;
    println!("Learning response: {}\n", &result3[..std::cmp::min(result3.len(), 300)]);

    // Test 4: Ask about the learned preference
    println!("=== Test 4: Recall Test ===");
    let task4 = Task::new(
        "What programming language do I prefer and why?".to_string(),
        Some("Answer based on remembered information".to_string()),
    );

    let result4 = memory_agent.call_with_user(task4, Some("samet".to_string()))
        .await.map_err(|e| anyhow::anyhow!(e))?;
    println!("Recall response: {}\n", &result4[..std::cmp::min(result4.len(), 300)]);


    // Test 5: Ask user name to test is agent remembers user name
    println!("=== Test 5: User Name Test ===");
    let task5 = Task::new(
        "What is my name?".to_string(),
        Some("Answer based on remembered information".to_string()),
    );
    
    let result5 = memory_agent.call_with_user(task5, Some("samet".to_string()))
        .await.map_err(|e| anyhow::anyhow!(e))?;
    println!("User name response: {}\n", &result5[..std::cmp::min(result5.len(), 300)]);

    // ================================
    // COMPARISON WITH NON-MEMORY AGENT
    // ================================
    println!("\n=== 🔄 Comparison: Agent Without Memory ===");
    
    let mut basic_agent = Agent::new_without_memory(
        agent_llm_config.clone(),
        "You are a helpful assistant.".to_string(),
        vec!["Provide accurate responses".to_string()],
        vec![]
    );

    println!("Basic agent memory stats: {:?}", basic_agent.get_memory_stats());

    let task5 = Task::new(
        "Can you give me an example like we discussed?".to_string(),
        Some("An example with no previous context".to_string()),
    );

    let result5 = basic_agent.call(task5).await.map_err(|e| anyhow::anyhow!(e))?;
    println!("Non-memory response: {}\n", &result5[..std::cmp::min(result5.len(), 300)]);

    // ================================
    // TRADITIONAL EXAMPLES (Updated)
    // ================================
    println!("=== 📊 JSON Validation with Memory ===");
    
    let json_task = Task::new_simple_json(
        "Create a user profile for someone named John who is 30 years old".to_string(),
        Some("User profile in JSON format".to_string()),
        vec![
            ("name".to_string(), JsonFieldType::String),
            ("age".to_string(), JsonFieldType::Number),
            ("active".to_string(), JsonFieldType::Boolean),
        ],
        true, // strict mode
    );

    match memory_agent.call_with_user(json_task, Some("demo_user".to_string())).await {
        Ok(result) => {
            println!("JSON Task Result: {}", result);
            // Parse and pretty-print the JSON
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&result) {
                println!("Parsed JSON: {}", serde_json::to_string_pretty(&parsed)?);
            }
        },
        Err(e) => println!("JSON Task Error: {}", e),
    }

    // ================================
    // TOOLS WITH MEMORY
    // ================================
    println!("\n=== 🛠️ Tools with Memory ===");
    let tools = get_tools_by_names(&["get_current_time"]);
    let mut tool_agent = Agent::with_memory(
        agent_llm_config.clone(),
        "You are a helpful assistant with access to tools and memory.".to_string(),
        vec!["Use tools when needed and remember tool usage patterns".to_string()],
        tools,
        AgentMemoryConfig::new()
    ).await.map_err(|e| anyhow::anyhow!("Failed to create tool agent: {}", e))?;

    let tool_task = Task::new(
        "What time is it right now?".to_string(),
        Some("Current time information".to_string()),
    );

    match tool_agent.call_with_user(tool_task, Some("demo_user".to_string())).await {
        Ok(result) => println!("Tool Task Result: {}", result),
        Err(e) => println!("Tool Task Error: {}", e),
    }

    // ================================
    // SUMMARY
    // ================================
    println!("\n🎯 Memory-Enhanced Agent Features Summary:");
    println!("┌─────────────────────────────────────────────────────────┐");
    println!("│ ✅ Automatic memory storage and retrieval              │");
    println!("│ ✅ User-specific context and personalization           │");
    println!("│ ✅ Learning from interactions (success & failure)      │");
    println!("│ ✅ Cross-session memory persistence                    │");
    println!("│ ✅ Configurable memory systems                         │");
    println!("│ ✅ Manual knowledge injection (facts & procedures)     │");
    println!("│ ✅ JSON validation with memory-driven improvements     │");
    println!("│ ✅ Tool usage with memory of patterns                  │");
    println!("│ ✅ Performance mode (memory can be disabled)           │");
    println!("└─────────────────────────────────────────────────────────┘");

    println!("\n💡 Key Benefits:");
    println!("• Agents remember user preferences and context");
    println!("• Continuous learning from every interaction");
    println!("• Personalized responses based on history");
    println!("• Intelligent context retrieval for better answers");
    println!("• Configurable for different use cases");

    Ok(())
}
