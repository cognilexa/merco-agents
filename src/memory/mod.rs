// Core memory module with reorganized structure
pub mod config;
pub mod types;
pub mod storage;
pub mod embedding;
pub mod manager;

// Legacy memory system (will be deprecated)
pub mod memory_manager;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Core memory types supported by the system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MemoryType {
    Working,    // Short-term conversation context
    Semantic,   // Factual knowledge and information
    Procedural, // How-to knowledge and processes
    Episodic,   // Past experiences and interactions
}

/// Memory entry structure with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub content: String,
    pub metadata: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
    pub memory_type: MemoryType,
    pub relevance_score: Option<f32>,
    pub embeddings: Option<Vec<f32>>,
}

/// Memory retrieval query
#[derive(Debug, Clone)]
pub struct MemoryQuery {
    pub content: String,
    pub memory_types: Vec<MemoryType>,
    pub max_results: usize,
    pub similarity_threshold: f32,
    pub metadata_filters: HashMap<String, String>,
}

/// Memory retrieval result
#[derive(Debug, Clone)]
pub struct MemoryResult {
    pub entries: Vec<MemoryEntry>,
    pub total_found: usize,
    pub search_time_ms: u64,
}

/// Core memory storage trait
#[async_trait]
pub trait MemoryStorage: Send + Sync {
    async fn store(&mut self, entry: MemoryEntry) -> Result<String, String>;
    async fn retrieve(&self, query: MemoryQuery) -> Result<MemoryResult, String>;
    async fn update(&mut self, id: &str, entry: MemoryEntry) -> Result<(), String>;
    async fn delete(&mut self, id: &str) -> Result<(), String>;
    async fn clear(&mut self, memory_type: Option<MemoryType>) -> Result<(), String>;
}

/// Working memory for conversation context
#[async_trait]
pub trait WorkingMemory: Send + Sync {
    async fn add_message(&mut self, role: String, content: String) -> Result<(), String>;
    async fn get_context(&self, max_tokens: usize) -> Result<String, String>;
    async fn summarize_old_context(&mut self) -> Result<(), String>;
    async fn clear(&mut self) -> Result<(), String>;
}

/// Semantic memory for factual knowledge
#[async_trait]
pub trait SemanticMemory: Send + Sync {
    async fn store_knowledge(&mut self, content: String, metadata: HashMap<String, String>) -> Result<String, String>;
    async fn search_knowledge(&self, query: &str, max_results: usize) -> Result<Vec<MemoryEntry>, String>;
    async fn update_knowledge(&mut self, id: &str, content: String) -> Result<(), String>;
}

/// Procedural memory for processes and how-to knowledge
#[async_trait]
pub trait ProceduralMemory: Send + Sync {
    async fn store_procedure(&mut self, name: String, steps: Vec<String>, metadata: HashMap<String, String>) -> Result<String, String>;
    async fn get_procedure(&self, name: &str) -> Result<Option<Vec<String>>, String>;
    async fn search_procedures(&self, query: &str) -> Result<Vec<MemoryEntry>, String>;
}

/// Episodic memory for experiences and interactions
#[async_trait]
pub trait EpisodicMemory: Send + Sync {
    async fn store_experience(&mut self, user_id: String, interaction: String, metadata: HashMap<String, String>) -> Result<String, String>;
    async fn get_user_history(&self, user_id: &str, max_results: usize) -> Result<Vec<MemoryEntry>, String>;
    async fn search_experiences(&self, query: &str, user_id: Option<String>) -> Result<Vec<MemoryEntry>, String>;
}

/// Memory consolidation for moving between memory types
#[async_trait]
pub trait MemoryConsolidation: Send + Sync {
    async fn consolidate_working_to_episodic(&mut self, user_id: String) -> Result<(), String>;
    async fn extract_knowledge_from_interactions(&mut self, user_id: String) -> Result<Vec<MemoryEntry>, String>;
    async fn identify_important_procedures(&mut self) -> Result<Vec<MemoryEntry>, String>;
}

// Re-export key types for easy access
pub use config::{MemoryConfig, EmbeddingProvider, StorageBackend, MemoryLimits};
pub use manager::{AgentMemory, AgentMemoryFactory};
pub use storage::{MetadataStorage, VectorStorage};
pub use embedding::EmbeddingProviderTrait; 