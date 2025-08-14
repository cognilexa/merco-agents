# Merco-Agents

A powerful Rust library for building intelligent AI agents with advanced memory capabilities, task execution, and tool integration.

## 🧠 **NEW: Memory-Enabled Agents**

Merco-Agents now features a comprehensive memory system that allows agents to:

- **Remember user interactions** across sessions
- **Learn from successes and failures** to improve over time
- **Provide personalized responses** based on user history  
- **Use multiple memory types** (Working, Semantic, Episodic, Procedural)
- **Automatically store and retrieve context** for better conversations

### Quick Start with Memory

```rust
use merco_agents::agent::agent::{Agent, AgentLLMConfig, AgentMemoryConfig};
use merco_agents::memory::config::{MemoryConfig, EmbeddingProvider, StorageBackend};

// Create an agent with memory capabilities
let memory_config = AgentMemoryConfig::new()
    .with_auto_store(true)     // Automatically remember interactions
    .with_auto_retrieve(true)  // Use memory for context
    .with_context_limit(5);    // Max memories per response

let mut agent = Agent::with_memory(
    llm_config,
    "You are a helpful assistant with memory".to_string(),
    vec!["Remember user preferences".to_string()],
    vec![], // tools
    memory_config
);

// The agent automatically remembers this interaction
let result = agent.call_with_user(task, Some("user_123".to_string())).await?;

// Later conversations will use previous context
let followup = agent.call_with_user(followup_task, Some("user_123".to_string())).await?;
```

### Memory System Features

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Working Memory** | Short-term conversation context | Chat continuity, current session |
| **Semantic Memory** | Facts and knowledge | User preferences, learned information |
| **Episodic Memory** | Past experiences and interactions | User history, previous conversations |
| **Procedural Memory** | Skills and processes | How-to knowledge, best practices |

### Memory Configuration Options

```rust
// Default configuration (recommended for most use cases)
let agent = Agent::new(llm_config, backstory, goals, tools);

// Performance mode (no memory)
let agent = Agent::new_without_memory(llm_config, backstory, goals, tools);

// Custom memory configuration
let memory_config = MemoryConfig {
    embedding: EmbeddingProvider::HuggingFace,  // or OpenAI, Ollama, Custom
    storage: StorageBackend::SQLiteInMemory,    // or PostgreSQL, Qdrant
    limits: MemoryLimits {
        max_working_memory_messages: 50,
        max_retrieval_results: 10,
        similarity_threshold: 0.7,
        // ... other settings
    },
};

let agent_config = AgentMemoryConfig::new()
    .with_memory_config(memory_config)
    .with_auto_store(true)
    .with_auto_retrieve(true);

let agent = Agent::with_memory(llm_config, backstory, goals, tools, agent_config);
```

### Manual Memory Management

```rust
// Store facts for the agent to remember
agent.learn_fact(
    "User prefers JSON responses over plain text".to_string(),
    Some("preferences".to_string()),
    Some(0.9) // importance score
).await?;

// Teach procedures
agent.learn_procedure(
    "How to format code examples".to_string(),
    vec![
        "Use proper syntax highlighting".to_string(),
        "Include comments explaining key parts".to_string(),
        "Provide a brief explanation".to_string(),
    ],
    Some("coding".to_string())
).await?;

// Store specific experiences
agent.store_memory(
    "User had trouble with async/await concepts".to_string(),
    MemoryType::Episodic,
    Some("user_123".to_string()),
    Some(metadata)
).await?;

// Query memories manually
let memories = agent.retrieve_memories(
    "async programming difficulties",
    Some("user_123".to_string()),
    "educational context"
).await?;
```

## 🚀 Features

- **Intelligent Agents**: Create AI agents with custom backstories, goals, and capabilities
- **Memory System**: Advanced memory capabilities with multiple storage backends
- **Task Execution**: Structured task handling with validation and retry logic
- **Tool Integration**: Easy integration with external tools and APIs
- **JSON Validation**: Robust JSON output validation with schema support
- **Crew Workflows**: Coordinate multiple agents in sequential or hierarchical workflows
- **Embedding Support**: Multiple embedding providers (OpenAI, Ollama, HuggingFace, Custom)
- **Storage Backends**: SQLite, PostgreSQL, Qdrant, and in-memory options

## Examples

The `examples/` directory contains comprehensive demonstrations:

- **`agent_with_memory.rs`**: Complete memory system showcase
- **`memory_demo.rs`**: Memory types and storage backends
- **`basic_agent/`**: Simple agent interactions  
- **`json_validation/`**: JSON output validation
- **`tool_usage/`**: Custom tool integration

Run examples with:
```bash
cargo run --example agent_with_memory
cargo run --example memory_demo
```

## Architecture

### Agent System
- **Agent**: Core agent with configurable memory capabilities
- **Task**: Structured task execution with validation
- **Tool**: External function integration
- **Crew**: Multi-agent coordination

### Memory System  
- **Memory Manager**: Intelligent memory storage and retrieval
- **Storage Backends**: Pluggable storage (SQLite, PostgreSQL, Qdrant)
- **Embedding Providers**: Multiple embedding options
- **Memory Types**: Working, Semantic, Episodic, Procedural

## Configuration

Set up your environment:

```bash
# Required
export OPENROUTER_API_KEY="your-api-key"

# Optional memory configurations
export SQLITE_PATH="./agent_memory.db"
export QDRANT_URL="http://localhost:6333" 
export OLLAMA_URL="http://localhost:11434"
```

## Benefits of Memory-Enabled Agents

✅ **Personalization**: Agents remember user preferences and adapt responses  
✅ **Continuous Learning**: Improve performance through experience  
✅ **Context Awareness**: Better conversations with relevant context  
✅ **User Relationships**: Build long-term relationships with users  
✅ **Error Learning**: Learn from mistakes to avoid future errors  
✅ **Knowledge Retention**: Preserve important information across sessions  

## Use Cases

- **Customer Support**: Remember customer history and preferences
- **Educational Tutors**: Track learning progress and adapt explanations  
- **Personal Assistants**: Learn user habits and preferences
- **Code Assistants**: Remember project context and coding patterns
- **Content Creation**: Maintain consistent voice and style preferences

This memory system transforms simple chatbots into intelligent agents that truly understand and adapt to their users! 🧠✨ 