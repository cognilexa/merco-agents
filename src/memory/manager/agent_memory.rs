use async_trait::async_trait;
use std::collections::HashMap;
use super::super::{MemoryEntry, MemoryType, MemoryResult};
use super::super::config::MemoryConfig;
use super::super::storage::{MetadataStorage, VectorStorage, create_metadata_storage, create_vector_storage};
use super::super::embedding::{EmbeddingProviderTrait, create_embedding_provider};

/// Agent-specific memory manager that integrates with persistent storage
pub struct AgentMemory {
    agent_id: String,
    user_id: Option<String>,
    metadata_storage: Box<dyn MetadataStorage>,
    vector_storage: Box<dyn VectorStorage>,
    embedding_provider: Box<dyn EmbeddingProviderTrait>,
    config: MemoryConfig,
}

impl AgentMemory {
    /// Create a new agent memory instance
    pub async fn new(
        agent_id: String,
        user_id: Option<String>,
        config: MemoryConfig,
    ) -> Result<Self, String> {
        let storage_config = config.storage_config();
        let embedding_config = config.embedding_config();
        
        let metadata_storage = create_metadata_storage(&storage_config)
            .await
            .map_err(|e| format!("Failed to create metadata storage: {}", e))?;
        
        let vector_storage = create_vector_storage(&storage_config)
            .await
            .map_err(|e| format!("Failed to create vector storage: {}", e))?;
        
        let embedding_provider = create_embedding_provider(&embedding_config)
            .map_err(|e| format!("Failed to create embedding provider: {}", e))?;

        Ok(Self {
            agent_id,
            user_id,
            metadata_storage,
            vector_storage,
            embedding_provider,
            config,
        })
    }

    /// Store a memory entry with automatic embedding generation
    pub async fn store_memory(
        &mut self,
        content: String,
        memory_type: MemoryType,
        metadata: HashMap<String, String>,
    ) -> Result<String, String> {
        // Generate embeddings
        let embeddings = self.embedding_provider
            .embed_text(&content)
            .await
            .map_err(|e| format!("Embedding generation failed: {}", e))?;

        // Create memory entry
        let mut entry_metadata = metadata;
        entry_metadata.insert("agent_id".to_string(), self.agent_id.clone());
        if let Some(ref user_id) = self.user_id {
            entry_metadata.insert("user_id".to_string(), user_id.clone());
        }

        let entry = MemoryEntry {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.clone(),
            metadata: entry_metadata.clone(),
            timestamp: chrono::Utc::now(),
            memory_type,
            relevance_score: Some(0.5),
            embeddings: Some(embeddings.clone()),
        };

        // Store metadata
        self.metadata_storage.store_metadata(&entry).await
            .map_err(|e| format!("Metadata storage failed: {}", e))?;

        // Store vector
        self.vector_storage.store_vector(&entry.id, &embeddings, entry_metadata).await
            .map_err(|e| format!("Vector storage failed: {}", e))?;

        Ok(entry.id)
    }

    /// Search memories using semantic similarity
    pub async fn search_memories(
        &self,
        query: &str,
        memory_types: Option<Vec<MemoryType>>,
        max_results: Option<usize>,
    ) -> Result<MemoryResult, String> {
        let start_time = std::time::Instant::now();
        
        // Generate query embedding
        let query_embedding = self.embedding_provider
            .embed_text(query)
            .await
            .map_err(|e| format!("Query embedding failed: {}", e))?;

        // Search vectors
        let vector_results = self.vector_storage
            .search_vectors(
                &query_embedding,
                max_results.unwrap_or(self.config.limits.max_retrieval_results),
                self.config.limits.similarity_threshold,
            )
            .await
            .map_err(|e| format!("Vector search failed: {}", e))?;

        // Get metadata for found vectors
        let mut entries = Vec::new();
        for vector_result in vector_results {
            if let Ok(Some(mut entry)) = self.metadata_storage.get_metadata(&vector_result.id).await {
                // Filter by memory type if specified
                if let Some(ref types) = memory_types {
                    if !types.contains(&entry.memory_type) {
                        continue;
                    }
                }
                
                // Filter by agent/user
                let matches_agent = entry.metadata.get("agent_id") == Some(&self.agent_id);
                let matches_user = if let Some(ref user_id) = self.user_id {
                    entry.metadata.get("user_id") == Some(user_id) || matches_agent
                } else {
                    matches_agent
                };

                if matches_user {
                    entry.relevance_score = Some(vector_result.score);
                    entries.push(entry);
                }
            }
        }

        let search_time = start_time.elapsed().as_millis() as u64;
        let total_found = entries.len();

        Ok(MemoryResult {
            entries,
            total_found,
            search_time_ms: search_time,
        })
    }

    /// Get agent-specific memories
    pub async fn get_agent_memories(&self, limit: usize) -> Result<Vec<MemoryEntry>, String> {
        // First try to get by agent_id
        let mut entries = Vec::new();
        
        // Search in metadata storage for agent-specific entries
        for memory_type in [MemoryType::Working, MemoryType::Semantic, MemoryType::Procedural, MemoryType::Episodic] {
            let type_entries = self.metadata_storage
                .list_by_type(memory_type, limit)
                .await
                .map_err(|e| format!("Failed to get memories: {}", e))?;
            
            // Filter by agent_id
            for entry in type_entries {
                if entry.metadata.get("agent_id") == Some(&self.agent_id) {
                    entries.push(entry);
                }
            }
            
            if entries.len() >= limit {
                break;
            }
        }

        entries.truncate(limit);
        Ok(entries)
    }

    /// Get user-related memories (shared across agents for the same user)
    pub async fn get_user_memories(&self, limit: usize) -> Result<Vec<MemoryEntry>, String> {
        if let Some(ref user_id) = self.user_id {
            self.metadata_storage
                .list_by_user(user_id, limit)
                .await
                .map_err(|e| format!("Failed to get user memories: {}", e))
        } else {
            Ok(Vec::new())
        }
    }

    /// Delete a memory entry
    pub async fn delete_memory(&mut self, memory_id: &str) -> Result<(), String> {
        // Delete from metadata storage
        self.metadata_storage.delete_metadata(memory_id).await
            .map_err(|e| format!("Failed to delete metadata: {}", e))?;

        // Delete from vector storage
        self.vector_storage.delete_vector(memory_id).await
            .map_err(|e| format!("Failed to delete vector: {}", e))?;

        Ok(())
    }

    /// Get agent ID
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Get user ID
    pub fn user_id(&self) -> Option<&str> {
        self.user_id.as_deref()
    }

    /// Update configuration
    pub fn update_config(&mut self, config: MemoryConfig) {
        self.config = config;
    }
}

/// Agent memory factory for creating agent-specific memory instances
pub struct AgentMemoryFactory {
    config: MemoryConfig,
}

impl AgentMemoryFactory {
    pub fn new(config: MemoryConfig) -> Self {
        Self { config }
    }

    /// Create memory for a specific agent
    pub async fn create_agent_memory(
        &self,
        agent_id: String,
        user_id: Option<String>,
    ) -> Result<AgentMemory, String> {
        AgentMemory::new(agent_id, user_id, self.config.clone()).await
    }

    /// Create memory with custom config
    pub async fn create_agent_memory_with_config(
        &self,
        agent_id: String,
        user_id: Option<String>,
        config: MemoryConfig,
    ) -> Result<AgentMemory, String> {
        AgentMemory::new(agent_id, user_id, config).await
    }
} 