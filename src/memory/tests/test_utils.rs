use super::super::*;
use std::env;
use tempfile::TempDir;

pub struct TestEnv {
    pub temp_dir: TempDir,
    pub config: MemoryConfig,
}

impl TestEnv {
    pub fn new() -> Self {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        
        // Set up test environment variables
        env::set_var("SQLITE_PATH", temp_dir.path().join("test.db").to_str().unwrap());
        env::set_var("OPENAI_API_KEY", "test-key");
        
        let config = MemoryConfig {
            embedding: EmbeddingProvider::OpenAI,
            storage: StorageBackend::SQLiteInMemory,
            limits: MemoryLimits::default(),
        };

        Self {
            temp_dir,
            config,
        }
    }

    pub fn with_embedding(mut self, provider: EmbeddingProvider) -> Self {
        self.config.embedding = provider;
        self
    }

    pub fn with_storage(mut self, backend: StorageBackend) -> Self {
        self.config.storage = backend;
        self
    }

    pub fn with_limits(mut self, limits: MemoryLimits) -> Self {
        self.config.limits = limits;
        self
    }
}

impl Drop for TestEnv {
    fn drop(&mut self) {
        env::remove_var("SQLITE_PATH");
        env::remove_var("OPENAI_API_KEY");
    }
}

// Mock embedding provider for tests
#[derive(Clone)]
pub struct MockEmbeddingProvider;

#[async_trait::async_trait]
impl EmbeddingProviderTrait for MockEmbeddingProvider {
    async fn embed_text(&self, _text: &str) -> Result<Vec<f32>, String> {
        Ok(vec![0.1, 0.2, 0.3])
    }
}

// Mock storage implementations
pub struct MockMetadataStorage;

#[async_trait::async_trait]
impl MetadataStorage for MockMetadataStorage {
    async fn store_metadata(&self, _entry: &MemoryEntry) -> Result<(), String> {
        Ok(())
    }

    async fn get_metadata(&self, _id: &str) -> Result<Option<MemoryEntry>, String> {
        Ok(None)
    }

    async fn delete_metadata(&self, _id: &str) -> Result<(), String> {
        Ok(())
    }

    async fn list_by_type(&self, _memory_type: MemoryType, _limit: usize) -> Result<Vec<MemoryEntry>, String> {
        Ok(vec![])
    }

    async fn list_by_user(&self, _user_id: &str, _limit: usize) -> Result<Vec<MemoryEntry>, String> {
        Ok(vec![])
    }
}

pub struct MockVectorStorage;

#[async_trait::async_trait]
impl VectorStorage for MockVectorStorage {
    async fn store_vector(
        &self,
        _id: &str,
        _vector: &[f32],
        _metadata: std::collections::HashMap<String, String>,
    ) -> Result<(), String> {
        Ok(())
    }

    async fn search_vectors(
        &self,
        _query_vector: &[f32],
        _limit: usize,
        _threshold: f32,
    ) -> Result<Vec<VectorSearchResult>, String> {
        Ok(vec![])
    }

    async fn delete_vector(&self, _id: &str) -> Result<(), String> {
        Ok(())
    }
} 