use super::super::manager::agent_memory::*;
use super::super::types::*;
use super::test_utils::*;
use std::collections::HashMap;
use tokio;

#[tokio::test]
async fn test_agent_memory_creation() {
    let test_env = TestEnv::new();
    
    let memory = AgentMemory::new(
        "test-agent".to_string(),
        Some("test-user".to_string()),
        test_env.config,
    ).await;
    
    assert!(memory.is_ok());
    let memory = memory.unwrap();
    assert_eq!(memory.agent_id(), "test-agent");
    assert_eq!(memory.user_id(), Some("test-user"));
}

#[tokio::test]
async fn test_store_and_retrieve_memory() {
    let test_env = TestEnv::new();
    
    let mut memory = AgentMemory::new(
        "test-agent".to_string(),
        Some("test-user".to_string()),
        test_env.config,
    ).await.unwrap();

    // Store a memory
    let content = "Test memory content".to_string();
    let mut metadata = HashMap::new();
    metadata.insert("test_key".to_string(), "test_value".to_string());

    let memory_id = memory.store_memory(
        content.clone(),
        MemoryType::Semantic,
        metadata.clone(),
    ).await.unwrap();

    // Search for the stored memory
    let results = memory.search_memories(
        "Test memory",
        Some(vec![MemoryType::Semantic]),
        Some(1),
    ).await.unwrap();

    assert!(results.total_found > 0);
    assert!(results.entries.iter().any(|e| e.content == content));
}

#[tokio::test]
async fn test_agent_specific_memories() {
    let test_env = TestEnv::new();
    
    let mut memory = AgentMemory::new(
        "test-agent".to_string(),
        Some("test-user".to_string()),
        test_env.config,
    ).await.unwrap();

    // Store multiple memories
    for i in 0..3 {
        let content = format!("Memory {}", i);
        let metadata = HashMap::new();
        memory.store_memory(
            content,
            MemoryType::Working,
            metadata,
        ).await.unwrap();
    }

    // Retrieve agent-specific memories
    let memories = memory.get_agent_memories(5).await.unwrap();
    assert_eq!(memories.len(), 3);
}

#[tokio::test]
async fn test_memory_deletion() {
    let test_env = TestEnv::new();
    
    let mut memory = AgentMemory::new(
        "test-agent".to_string(),
        None,
        test_env.config,
    ).await.unwrap();

    // Store a memory
    let memory_id = memory.store_memory(
        "Delete test".to_string(),
        MemoryType::Episodic,
        HashMap::new(),
    ).await.unwrap();

    // Delete the memory
    let delete_result = memory.delete_memory(&memory_id).await;
    assert!(delete_result.is_ok());

    // Verify deletion
    let search_results = memory.search_memories(
        "Delete test",
        Some(vec![MemoryType::Episodic]),
        Some(1),
    ).await.unwrap();
    
    assert_eq!(search_results.total_found, 0);
}

#[tokio::test]
async fn test_memory_factory() {
    let test_env = TestEnv::new();
    let factory = AgentMemoryFactory::new(test_env.config);

    // Create memory instances
    let memory1 = factory.create_agent_memory(
        "agent1".to_string(),
        Some("user1".to_string()),
    ).await;
    
    let memory2 = factory.create_agent_memory(
        "agent2".to_string(),
        Some("user2".to_string()),
    ).await;

    assert!(memory1.is_ok());
    assert!(memory2.is_ok());
    assert_ne!(memory1.unwrap().agent_id(), memory2.unwrap().agent_id());
} 