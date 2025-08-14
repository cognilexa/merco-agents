use super::super::config::*;
use std::env;

#[test]
fn test_default_config() {
    let config = MemoryConfig::default();
    assert_eq!(config.embedding, EmbeddingProvider::OpenAI);
    assert_eq!(config.storage, StorageBackend::SQLiteInMemory);
    
    let limits = config.limits;
    assert_eq!(limits.max_working_memory_messages, 50);
    assert_eq!(limits.max_retrieval_results, 10);
    assert_eq!(limits.similarity_threshold, 0.7);
    assert_eq!(limits.importance_threshold, 0.3);
    assert_eq!(limits.consolidation_interval_hours, 1);
}

#[test]
fn test_config_from_env() {
    // Test OpenAI configuration
    env::set_var("OPENAI_API_KEY", "test-key");
    let config = MemoryConfig::from_env();
    assert_eq!(config.embedding, EmbeddingProvider::OpenAI);
    
    // Test Ollama configuration
    env::set_var("OLLAMA_URL", "http://localhost:11434");
    let config = MemoryConfig::from_env();
    assert_eq!(config.embedding, EmbeddingProvider::Ollama);
    
    // Test PostgreSQL configuration
    env::set_var("DATABASE_URL", "postgres://user:pass@localhost/db");
    let config = MemoryConfig::from_env();
    assert_eq!(config.storage, StorageBackend::PostgreSQLInMemory);
    
    // Cleanup
    env::remove_var("OPENAI_API_KEY");
    env::remove_var("OLLAMA_URL");
    env::remove_var("DATABASE_URL");
}

#[test]
fn test_embedding_config() {
    let config = MemoryConfig::default();
    let embedding_config = config.embedding_config();
    
    assert_eq!(embedding_config.provider_type, "openai");
    assert_eq!(embedding_config.model, "text-embedding-3-small");
    assert_eq!(embedding_config.dimension, 1536);
}

#[test]
fn test_storage_config() {
    let config = MemoryConfig::default();
    let storage_config = config.storage_config();
    
    assert_eq!(storage_config.metadata_type, "sqlite");
    assert_eq!(storage_config.vector_type, "memory");
    assert_eq!(storage_config.collection_name, "memory");
} 