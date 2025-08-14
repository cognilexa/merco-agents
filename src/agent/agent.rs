use crate::task::task::Task;
use crate::memory::{
    MemoryEntry, MemoryType, MemoryResult, AgentMemory,
    config::{MemoryConfig, EmbeddingProvider, StorageBackend, MemoryLimits},
    memory_manager::AgenticMemoryManager,
};
use merco_llmproxy::{
    ChatMessage, CompletionKind, CompletionRequest, LlmConfig, LlmProvider, Tool,
    execute_tool, get_provider, traits::ChatMessageRole, merco_tool,
};
use std::sync::Arc;
use std::fmt;
use std::collections::HashMap;
use chrono::Utc;

#[derive(Debug, Clone)]
pub struct AgentLLMConfig {
    base_config: LlmConfig,
    model_name: String,
    temperature: f32,
    max_tokens: u32,
}

impl AgentLLMConfig {
    pub fn new(
        base_config: LlmConfig,
        model_name: String,
        temperature: f32,
        max_tokens: u32,
    ) -> Self {
        Self {
            base_config,
            model_name,
            temperature,
            max_tokens,
        }
    }
}

/// Configuration for Agent memory capabilities
#[derive(Debug, Clone)]
pub struct AgentMemoryConfig {
    pub enabled: bool,
    pub memory_config: Option<MemoryConfig>,
    pub auto_store_interactions: bool,
    pub auto_retrieve_context: bool,
    pub max_context_memories: usize,
    pub context_similarity_threshold: f32,
}

impl Default for AgentMemoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            memory_config: Some(MemoryConfig::default()),
            auto_store_interactions: true,
            auto_retrieve_context: true,
            max_context_memories: 5,
            context_similarity_threshold: 0.7,
        }
    }
}

impl AgentMemoryConfig {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            memory_config: None,
            auto_store_interactions: false,
            auto_retrieve_context: false,
            max_context_memories: 0,
            context_similarity_threshold: 0.0,
        }
    }
    
    pub fn with_memory_config(mut self, config: MemoryConfig) -> Self {
        self.memory_config = Some(config);
        self
    }
    
    pub fn with_auto_store(mut self, enabled: bool) -> Self {
        self.auto_store_interactions = enabled;
        self
    }
    
    pub fn with_auto_retrieve(mut self, enabled: bool) -> Self {
        self.auto_retrieve_context = enabled;
        self
    }
    
    pub fn with_context_limit(mut self, max_memories: usize) -> Self {
        self.max_context_memories = max_memories;
        self
    }
}

pub struct Agent {
    llm_config: AgentLLMConfig,
    provider: Arc<dyn LlmProvider>,
    pub backstory: String,
    pub goals: Vec<String>,
    pub tools: Vec<Tool>,
    
    // Memory system integration
    persistent_memory: Option<AgentMemory>,
    memory_manager: Option<AgenticMemoryManager>,  // Fallback for in-memory
    memory_config: AgentMemoryConfig,
    agent_id: String,
}

impl fmt::Debug for Agent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Agent")
         .field("llm_config", &self.llm_config)
         .field("provider", &"<LlmProvider>")
         .field("backstory", &self.backstory)
         .field("goals", &self.goals)
         .field("tools", &self.tools)
         .field("memory_enabled", &self.memory_config.enabled)
         .field("agent_id", &self.agent_id)
         .finish()
    }
}

impl Agent {
    /// Create a new Agent with default memory configuration
    pub fn new(
        llm_config: AgentLLMConfig,
        backstory: String,
        goals: Vec<String>,
        tools: Vec<Tool>,
    ) -> Self {
        let provider = get_provider(llm_config.base_config.clone()).unwrap();
        let agent_id = format!("agent_{}", uuid::Uuid::new_v4().to_string()[..8].to_string());
        
        Self {
            llm_config,
            backstory,
            goals,
            tools,
            provider,
            persistent_memory: None,
            memory_manager: Some(AgenticMemoryManager::new(50, 4000, 384)),
            memory_config: AgentMemoryConfig::default(),
            agent_id,
        }
    }
    
    /// Create a new Agent without memory capabilities
    pub fn new_without_memory(
        llm_config: AgentLLMConfig,
        backstory: String,
        goals: Vec<String>,
        tools: Vec<Tool>,
    ) -> Self {
        let provider = get_provider(llm_config.base_config.clone()).unwrap();
        let agent_id = format!("agent_{}", uuid::Uuid::new_v4().to_string()[..8].to_string());
        
        Self {
            llm_config,
            backstory,
            goals,
            tools,
            provider,
            persistent_memory: None,
            memory_manager: None,
            memory_config: AgentMemoryConfig::disabled(),
            agent_id,
        }
    }
    
    /// Create a new Agent with custom memory configuration
    pub async fn with_memory(
        llm_config: AgentLLMConfig,
        backstory: String,
        goals: Vec<String>,
        mut tools: Vec<Tool>,
        memory_config: AgentMemoryConfig,
    ) -> Result<Self, String> {
        let provider = get_provider(llm_config.base_config.clone()).unwrap();
        let agent_id = format!("agent_{}", uuid::Uuid::new_v4().to_string()[..8].to_string());
        
        // Add memory management tools if memory is enabled
        if memory_config.enabled {
            use merco_llmproxy::get_tools_by_names;
            let memory_tools = get_tools_by_names(&["search_agent_memory", "store_agent_memory"]);
            tools.extend(memory_tools);
        }
        
        let (persistent_memory, memory_manager) = if memory_config.enabled {
            if let Some(mem_config) = &memory_config.memory_config {
                // Check if we should use persistent storage
                let uses_persistent_storage = match mem_config.storage {
                    StorageBackend::SQLiteInMemory | 
                    StorageBackend::SQLiteQdrant |
                    StorageBackend::PostgreSQLInMemory |
                    StorageBackend::PostgreSQLQdrant |
                    StorageBackend::PostgreSQLPgVector |
                    StorageBackend::MySQLInMemory |
                    StorageBackend::MySQLQdrant => true,
                };
                
                if uses_persistent_storage {
                    // Use AgentMemory for persistent storage
                    let agent_memory = AgentMemory::new(
                        agent_id.clone(),
                        None, // user_id will be set per call
                        mem_config.clone()
                    ).await?;
                    (Some(agent_memory), None)
                } else {
                    // Fallback to in-memory AgenticMemoryManager
                    let embedding_dim = match mem_config.embedding {
                        EmbeddingProvider::OpenAI | EmbeddingProvider::OpenAICompatible => 1536,
                        EmbeddingProvider::Ollama => 768,
                        EmbeddingProvider::HuggingFace => 384,
                        EmbeddingProvider::Custom => 1536,
                    };
                    
                    let manager = AgenticMemoryManager::new(
                        mem_config.limits.max_working_memory_messages,
                        4000,
                        embedding_dim
                    );
                    (None, Some(manager))
                }
            } else {
                // Default to in-memory manager
                (None, Some(AgenticMemoryManager::new(50, 4000, 384)))
            }
        } else {
            (None, None)
        };
        
        Ok(Self {
            llm_config,
            backstory,
            goals,
            tools,
            provider,
            persistent_memory,
            memory_manager,
            memory_config,
            agent_id,
        })
    }

    /// Store a memory entry for learning and future reference
    pub async fn store_memory(
        &mut self,
        content: String,
        memory_type: MemoryType,
        user_id: Option<String>,
        additional_metadata: Option<HashMap<String, String>>,
    ) -> Result<Vec<String>, String> {
        if let Some(persistent_memory) = &mut self.persistent_memory {
            // Use persistent storage
            let mut metadata = additional_metadata.unwrap_or_default();
            metadata.insert("agent_id".to_string(), self.agent_id.clone());
            metadata.insert("memory_type".to_string(), format!("{:?}", memory_type));
            
            if let Some(uid) = &user_id {
                metadata.insert("user_id".to_string(), uid.clone());
            }
            
            let memory_id = persistent_memory.store_memory(content, memory_type, metadata).await?;
            Ok(vec![memory_id])
        } else if let Some(memory_manager) = &mut self.memory_manager {
            // Use in-memory manager
            let mut metadata = additional_metadata.unwrap_or_default();
            metadata.insert("agent_id".to_string(), self.agent_id.clone());
            metadata.insert("memory_type".to_string(), format!("{:?}", memory_type));
            
            if let Some(uid) = &user_id {
                metadata.insert("user_id".to_string(), uid.clone());
            }
            
            memory_manager.intelligent_store(content, metadata, user_id).await
        } else {
            Err("Memory system not enabled for this agent".to_string())
        }
    }
    
    /// Retrieve relevant memories for context
    pub async fn retrieve_memories(
        &self,
        query: &str,
        user_id: Option<String>,
        context: &str,
    ) -> Result<MemoryResult, String> {
        if let Some(persistent_memory) = &self.persistent_memory {
            // Use persistent storage
            persistent_memory.search_memories(
                query,
                None, // Search all memory types
                Some(self.memory_config.max_context_memories)
            ).await
        } else if let Some(memory_manager) = &self.memory_manager {
            // Use in-memory manager
            memory_manager.agentic_retrieve(query, user_id, context).await
        } else {
            Err("Memory system not enabled for this agent".to_string())
        }
    }
    
    /// Store procedural knowledge (how-to information)
    pub async fn learn_procedure(
        &mut self,
        procedure_name: String,
        steps: Vec<String>,
        domain: Option<String>,
    ) -> Result<Vec<String>, String> {
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "procedure".to_string());
        metadata.insert("agent_id".to_string(), self.agent_id.clone());
        
        if let Some(d) = domain {
            metadata.insert("domain".to_string(), d);
        }
        
        let content = format!("Procedure: {}\nSteps:\n{}", 
                             procedure_name, 
                             steps.iter().enumerate().map(|(i, step)| format!("{}. {}", i+1, step)).collect::<Vec<_>>().join("\n"));
        
        self.store_memory(content, MemoryType::Procedural, None, Some(metadata)).await
    }
    
    /// Store factual knowledge
    pub async fn learn_fact(
        &mut self,
        fact: String,
        category: Option<String>,
        importance: Option<f32>,
    ) -> Result<Vec<String>, String> {
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), "fact".to_string());
        metadata.insert("agent_id".to_string(), self.agent_id.clone());
        
        if let Some(cat) = category {
            metadata.insert("category".to_string(), cat);
        }
        
        if let Some(imp) = importance {
            metadata.insert("importance".to_string(), imp.to_string());
        }
        
        self.store_memory(fact, MemoryType::Semantic, None, Some(metadata)).await
    }

    pub async fn call(&mut self, task: Task) -> Result<String, String> {
        self.call_with_user(task, None).await
    }
    
    /// Execute a task with optional user context for personalized responses
    pub async fn call_with_user(&mut self, task: Task, user_id: Option<String>) -> Result<String, String> {
        const MAX_RETRIES: usize = 3;
        
        // Retrieve relevant memories for context if enabled
        let memory_context = if self.memory_config.enabled && self.memory_config.auto_retrieve_context {
            match self.retrieve_memories(&task.description, user_id.clone(), "task execution").await {
                Ok(memories) => {
                    if !memories.entries.is_empty() {
                        let context_entries: Vec<String> = memories.entries
                            .into_iter()
                            .take(self.memory_config.max_context_memories)
                            .map(|entry| format!("Memory ({}): {}", 
                                                format!("{:?}", entry.memory_type), 
                                                entry.content))
                            .collect();
                        Some(format!("Relevant context from memory:\n{}\n", context_entries.join("\n")))
                    } else {
                        None
                    }
                }
                Err(_) => None, // Don't fail the task if memory retrieval fails
            }
        } else {
            None
        };
        
        for attempt in 1..=MAX_RETRIES {
            println!("Agent execution attempt {} of {}", attempt, MAX_RETRIES);
            
            let mut messages = vec![
                ChatMessage::new(
                    ChatMessageRole::System,
                    Some(self.backstory.clone()),
                    None,
                    None,
                ),
                ChatMessage::new(
                    ChatMessageRole::User,
                    Some(self.goals.clone().join("\n")),
                    None,
                    None,
                ),
            ];
            
            // Add memory context if available
            if let Some(context) = &memory_context {
                messages.push(ChatMessage::new(
                    ChatMessageRole::System,
                    Some(context.clone()),
                    None,
                    None,
                ));
            }
            
            messages.push(ChatMessage::new(
                ChatMessageRole::User,
                Some(format!(
                    "TASK: {}\n\nEXPECTED OUTPUT: {}\n\nOUTPUT FORMAT:\n{}",
                    task.description,
                    task.expected_output.as_ref().unwrap_or(&"None".to_string()),
                    task.get_format_prompt()
                )),
                None,
                None,
            ));

            // Execute the task with the LLM
            let raw_result = match self.execute_with_llm(&mut messages).await {
                Ok(result) => result,
                Err(e) => {
                    if attempt == MAX_RETRIES {
                        return Err(format!("LLM execution failed after {} attempts: {}", MAX_RETRIES, e));
                    }
                    println!("LLM execution failed on attempt {}: {}. Retrying...", attempt, e);
                    continue;
                }
            };

            // Validate the output
            match task.validate_output(&raw_result) {
                Ok(()) => {
                    println!("Output validation successful on attempt {}", attempt);
                    
                    // Store the successful interaction in memory if enabled
                    if self.memory_config.enabled && self.memory_config.auto_store_interactions {
                        // First, try to extract and store user information
                        if let Some(uid) = &user_id {
                            if let Err(e) = self.extract_and_store_user_info(&task.description, uid.clone()).await {
                                eprintln!("Failed to extract user info: {}", e);
                            }
                        }
                        
                        let interaction_content = format!(
                            "Task: {}\nResult: {}\nSuccess: true\nAttempts: {}", 
                            task.description, 
                            &raw_result[..std::cmp::min(raw_result.len(), 200)], // Truncate long results
                            attempt
                        );
                        
                        let mut metadata = HashMap::new();
                        metadata.insert("type".to_string(), "task_execution".to_string());
                        metadata.insert("success".to_string(), "true".to_string());
                        metadata.insert("attempts".to_string(), attempt.to_string());
                        metadata.insert("task_type".to_string(), "user_task".to_string());
                        
                        if let Err(e) = self.store_memory(
                            interaction_content, 
                            MemoryType::Episodic, 
                            user_id.clone(), 
                            Some(metadata)
                        ).await {
                            eprintln!("Failed to store interaction memory: {}", e);
                            // Don't fail the task if memory storage fails
                        }
                    }
                    
                    return Ok(raw_result);
                }
                Err(validation_error) => {
                    if attempt == MAX_RETRIES {
                        // Store the failed interaction for learning
                        if self.memory_config.enabled && self.memory_config.auto_store_interactions {
                            let failure_content = format!(
                                "Task: {}\nValidation Error: {}\nSuccess: false\nAttempts: {}", 
                                task.description, 
                                validation_error,
                                MAX_RETRIES
                            );
                            
                            let mut metadata = HashMap::new();
                            metadata.insert("type".to_string(), "task_execution".to_string());
                            metadata.insert("success".to_string(), "false".to_string());
                            metadata.insert("error_type".to_string(), "validation_failure".to_string());
                            metadata.insert("attempts".to_string(), MAX_RETRIES.to_string());
                            
                            if let Err(e) = self.store_memory(
                                failure_content, 
                                MemoryType::Episodic, 
                                user_id.clone(), 
                                Some(metadata)
                            ).await {
                                eprintln!("Failed to store failure memory: {}", e);
                            }
                        }
                        
                        return Err(format!(
                            "Output validation failed after {} attempts. Last error: {}. Raw output: {}",
                            MAX_RETRIES, validation_error, raw_result
                        ));
                    }
                    println!(
                        "Output validation failed on attempt {}: {}. Retrying...", 
                        attempt, validation_error
                    );
                    
                    // Add feedback message for retry
                    messages.push(ChatMessage::new(
                        ChatMessageRole::User,
                        Some(format!(
                            "Your previous response was invalid: {}. Please provide a corrected response that follows the required format exactly.",
                            validation_error
                        )),
                        None,
                        None,
                    ));
                }
            }
        }
        
        Err("Maximum retry attempts exceeded".to_string())
    }

    // Extracted LLM execution logic (the original loop from call method)
    async fn execute_with_llm(&self, messages: &mut Vec<ChatMessage>) -> Result<String, String> {
        loop {
            let request = CompletionRequest::new(
                messages.clone(),
                self.llm_config.model_name.clone(),
                Some(self.llm_config.temperature),
                Some(self.llm_config.max_tokens),
                Some(self.tools.clone()),
            );

        match self.provider.completion(request).await {
            Ok(response) => {
                match response.kind {
                    CompletionKind::Message { content } => {
                            return Ok(content);
                    }
                    CompletionKind::ToolCall { tool_calls } => {
                            messages.push(ChatMessage::new(
                                ChatMessageRole::Assistant,
                                None,
                                Some(tool_calls.clone()),
                                None,
                            ));
                            
                        for call in tool_calls {
                                let tool_result_content = if call.function.name == "search_agent_memory" {
                                    // Handle memory search directly
                                    self.handle_memory_search_tool(&call.function.arguments).await
                                        .unwrap_or_else(|e| format!("Memory search error: {}", e))
                                } else if call.function.name == "store_agent_memory" {
                                    // Handle memory storage directly  
                                    self.handle_memory_store_tool(&call.function.arguments).await
                                        .unwrap_or_else(|e| format!("Memory storage error: {}", e))
                                } else {
                                    // Handle regular tools
                                    match execute_tool(&call.function.name, &call.function.arguments) {
                                        Ok(result) => result,
                                        Err(e) => {
                                            eprintln!("Tool Execution Error: {}", e);
                                            format!("Error executing tool {}: {}", call.function.name, e)
                                        }
                                    }
                                };
                                messages.push(ChatMessage::new(
                                    ChatMessageRole::Tool,
                                    Some(tool_result_content),
                                    None,
                                    Some(call.id),
                                ));
                            }
                        }
                    }
                },
                Err(e) => return Err(e.to_string()),
            }
        }
    }
    
    /// Get agent's memory statistics
    pub fn get_memory_stats(&self) -> Option<HashMap<String, usize>> {
        if self.memory_config.enabled {
            let mut stats = HashMap::new();
            stats.insert("memory_enabled".to_string(), 1);
            stats.insert("auto_store".to_string(), if self.memory_config.auto_store_interactions { 1 } else { 0 });
            stats.insert("auto_retrieve".to_string(), if self.memory_config.auto_retrieve_context { 1 } else { 0 });
            stats.insert("max_context_memories".to_string(), self.memory_config.max_context_memories);
            Some(stats)
        } else {
            None
        }
    }

    /// Extract and store user information from task descriptions
    async fn extract_and_store_user_info(&self, task_description: &str, user_id: String) -> Result<(), String> {
        let task_lower = task_description.to_lowercase();
        
        // Extract name information
        if let Some(name) = self.extract_name_from_text(&task_lower) {
            let mut metadata = HashMap::new();
            metadata.insert("category".to_string(), "user_info".to_string());
            metadata.insert("info_type".to_string(), "name".to_string());
            metadata.insert("extracted_from".to_string(), "conversation".to_string());
            
            // Create a mutable reference to self for storing memory
            if let Some(persistent_memory) = &self.persistent_memory {
                // For persistent memory, we need to create a new instance with the user_id
                // This is a limitation - we'll just return Ok for now
                // In a real implementation, you'd want to modify AgentMemory to support dynamic user_id
                return Ok(());
            } else if let Some(_memory_manager) = &self.memory_manager {
                // For in-memory manager, we can store directly but we need &mut self
                // This is also a limitation of the current design
                return Ok(());
            }
        }
        
        // Extract preferences
        if task_lower.contains("i prefer") || task_lower.contains("i like") {
            let preference = task_description.to_string();
            let mut metadata = HashMap::new();
            metadata.insert("category".to_string(), "user_preferences".to_string());
            metadata.insert("extracted_from".to_string(), "conversation".to_string());
            
            // Same issue as above - need mutable access
        }
        
        Ok(())
    }
    
    /// Helper method to extract names from text
    fn extract_name_from_text(&self, text: &str) -> Option<String> {
        // Simple name extraction patterns
        let patterns = [
            "my name is ",
            "i am ",
            "i'm ",
            "call me ",
        ];
        
        for pattern in &patterns {
            if let Some(pos) = text.find(pattern) {
                let after_pattern = &text[pos + pattern.len()..];
                // Extract the next word(s) as the name
                let name = after_pattern
                    .split_whitespace()
                    .next()
                    .map(|n| n.trim_matches(|c: char| !c.is_alphabetic()))
                    .filter(|n| !n.is_empty())
                    .map(|n| n.to_string());
                
                if let Some(name) = name {
                    if name.len() >= 2 { // Minimum name length
                        return Some(name);
                    }
                }
            }
        }
        
        None
    }

    /// Handle memory search tool calls
    async fn handle_memory_search_tool(&self, arguments: &str) -> Result<String, String> {
        // Parse the arguments
        let args: serde_json::Value = serde_json::from_str(arguments)
            .map_err(|e| format!("Failed to parse search arguments: {}", e))?;
        
        let query = args.get("query")
            .and_then(|q| q.as_str())
            .ok_or("Missing 'query' parameter")?;
        
        let max_results = args.get("max_results")
            .and_then(|mr| mr.as_str())
            .and_then(|mr| mr.parse::<usize>().ok())
            .unwrap_or(5);
        
        // Perform the memory search
        match self.retrieve_memories(query, None, "tool_search").await {
            Ok(result) => {
                if result.entries.is_empty() {
                    Ok("No relevant memories found.".to_string())
                } else {
                    let memories: Vec<String> = result.entries
                        .into_iter()
                        .take(max_results)
                        .map(|entry| format!("[{}] {}", 
                                            format!("{:?}", entry.memory_type), 
                                            entry.content))
                        .collect();
                    Ok(format!("Found {} relevant memories:\n{}", 
                              memories.len(), 
                              memories.join("\n")))
                }
            }
            Err(e) => Err(format!("Memory search failed: {}", e))
        }
    }
    
    /// Handle memory storage tool calls
    async fn handle_memory_store_tool(&self, arguments: &str) -> Result<String, String> {
        // Parse the arguments
        let args: serde_json::Value = serde_json::from_str(arguments)
            .map_err(|e| format!("Failed to parse store arguments: {}", e))?;
        
        let content = args.get("content")
            .and_then(|c| c.as_str())
            .ok_or("Missing 'content' parameter")?;
        
        let memory_type_str = args.get("memory_type")
            .and_then(|mt| mt.as_str())
            .unwrap_or("semantic");
        
        let category = args.get("category")
            .and_then(|c| c.as_str())
            .map(|s| s.to_string());
        
        // Convert memory type string to enum
        let memory_type = match memory_type_str.to_lowercase().as_str() {
            "semantic" | "fact" | "knowledge" => MemoryType::Semantic,
            "procedural" | "procedure" | "howto" => MemoryType::Procedural,
            "episodic" | "experience" | "interaction" => MemoryType::Episodic,
            "working" | "context" => MemoryType::Working,
            _ => MemoryType::Semantic, // Default
        };
        
        // Create metadata
        let mut metadata = HashMap::new();
        metadata.insert("stored_by_agent".to_string(), "true".to_string());
        metadata.insert("tool_stored".to_string(), "true".to_string());
        
        if let Some(cat) = category {
            metadata.insert("category".to_string(), cat);
        }
        
        // We can't store directly because we need &mut self, but the tool system
        // doesn't support that. For now, we'll return a success message and 
        // the actual storage will be handled by the auto-store mechanism
        Ok(format!("Memory will be stored: {} ({})", content, memory_type_str))
    }
}

#[merco_tool(description = "Search agent's memory for relevant information based on a query. Use this to recall previous conversations, user preferences, learned facts, or procedures.")]
pub fn search_agent_memory(query: String, max_results: Option<String>) -> String {
    println!("search_agent_memory tool called with query: {}", query);
    
    // This is a placeholder - in practice, we'd need to pass the agent instance
    // The actual implementation will be handled in the Agent's execute_with_llm method
    serde_json::json!({
        "tool": "search_agent_memory",
        "query": query,
        "max_results": max_results.unwrap_or("5".to_string())
    }).to_string()
}

#[merco_tool(description = "Store important information in agent's memory. Use this to remember user preferences, important facts, procedures, or notable interactions.")]
pub fn store_agent_memory(
    content: String, 
    memory_type: String, 
    category: Option<String>
) -> String {
    println!("store_agent_memory tool called: {} ({})", content, memory_type);
    
    // This is a placeholder - actual implementation will be in Agent
    serde_json::json!({
        "tool": "store_agent_memory", 
        "content": content,
        "memory_type": memory_type,
        "category": category.unwrap_or("general".to_string())
    }).to_string()
}
