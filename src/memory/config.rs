use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Simple memory configuration - just specify what you want to use
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Embedding provider - simple enum selection
    pub embedding: EmbeddingProvider,
    /// Storage backend - simple enum selection  
    pub storage: StorageBackend,
    /// Optional limits and thresholds
    pub limits: MemoryLimits,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            embedding: EmbeddingProvider::OpenAI,
            storage: StorageBackend::SQLiteInMemory,
            limits: MemoryLimits::default(),
        }
    }
}

/// Simple embedding provider selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingProvider {
    /// OpenAI embeddings (uses OPENAI_API_KEY env var)
    OpenAI,
    /// OpenAI compatible (uses OPENAI_API_KEY and OPENAI_BASE_URL env vars)
    OpenAICompatible,
    /// Ollama local embeddings (uses OLLAMA_URL and OLLAMA_MODEL env vars, defaults: http://localhost:11434, all-minilm)
    Ollama,
    /// HuggingFace local models (uses HUGGINGFACE_MODEL env var, default: sentence-transformers/all-MiniLM-L6-v2)
    HuggingFace,
    /// Custom endpoint (uses CUSTOM_EMBEDDING_URL and optional CUSTOM_EMBEDDING_* env vars)
    Custom,
}

/// Simple storage backend selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageBackend {
    /// SQLite file + in-memory vectors (default, works out of box)
    SQLiteInMemory,
    /// SQLite file + Qdrant vectors (uses QDRANT_URL, QDRANT_API_KEY env vars)
    SQLiteQdrant,
    /// PostgreSQL + in-memory vectors (uses DATABASE_URL env var)
    PostgreSQLInMemory,
    /// PostgreSQL + Qdrant vectors (uses DATABASE_URL, QDRANT_URL, QDRANT_API_KEY env vars)
    PostgreSQLQdrant,
    /// PostgreSQL + pgvector (uses DATABASE_URL env var)
    PostgreSQLPgVector,
    /// MySQL + in-memory vectors (uses DATABASE_URL env var)
    MySQLInMemory,
    /// MySQL + Qdrant vectors (uses DATABASE_URL, QDRANT_URL, QDRANT_API_KEY env vars)
    MySQLQdrant,
}

/// Memory limits and thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLimits {
    pub max_working_memory_messages: usize,
    pub max_retrieval_results: usize,
    pub similarity_threshold: f32,
    pub importance_threshold: f32,
    pub consolidation_interval_hours: u64,
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            max_working_memory_messages: 50,
            max_retrieval_results: 10,
            similarity_threshold: 0.7,
            importance_threshold: 0.3,
            consolidation_interval_hours: 1,
        }
    }
}

impl MemoryConfig {
    /// Create config from environment variables - works out of the box
    pub fn from_env() -> Self {
        let mut config = Self::default();
        
        // Auto-detect embedding provider based on env vars
        if std::env::var("OLLAMA_URL").is_ok() || std::env::var("OLLAMA_MODEL").is_ok() {
            config.embedding = EmbeddingProvider::Ollama;
        } else if std::env::var("HUGGINGFACE_MODEL").is_ok() {
            config.embedding = EmbeddingProvider::HuggingFace;
        } else if std::env::var("CUSTOM_EMBEDDING_URL").is_ok() {
            config.embedding = EmbeddingProvider::Custom;
        } else if std::env::var("OPENAI_BASE_URL").is_ok() {
            config.embedding = EmbeddingProvider::OpenAICompatible;
        }
        // Default to OpenAI if OPENAI_API_KEY is available
        
        // Auto-detect storage based on env vars
        if std::env::var("DATABASE_URL").is_ok() {
            if std::env::var("QDRANT_URL").is_ok() {
                if std::env::var("DATABASE_URL").unwrap_or_default().contains("postgres") {
                    config.storage = StorageBackend::PostgreSQLQdrant;
                } else if std::env::var("DATABASE_URL").unwrap_or_default().contains("mysql") {
                    config.storage = StorageBackend::MySQLQdrant;
                }
            } else {
                if std::env::var("DATABASE_URL").unwrap_or_default().contains("postgres") {
                    config.storage = StorageBackend::PostgreSQLInMemory;
                } else if std::env::var("DATABASE_URL").unwrap_or_default().contains("mysql") {
                    config.storage = StorageBackend::MySQLInMemory;
                }
            }
        } else if std::env::var("QDRANT_URL").is_ok() {
            config.storage = StorageBackend::SQLiteQdrant;
        }
        // Default to SQLiteInMemory
        
        config
    }
    
    /// Get embedding configuration details
    pub fn embedding_config(&self) -> EmbeddingConfig {
        match &self.embedding {
            EmbeddingProvider::OpenAI => EmbeddingConfig {
                provider_type: "openai".to_string(),
                api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
                base_url: "https://api.openai.com/v1".to_string(),
                model: "text-embedding-3-small".to_string(),
                dimension: 1536,
                headers: HashMap::new(),
            },
            EmbeddingProvider::OpenAICompatible => EmbeddingConfig {
                provider_type: "openai".to_string(),
                api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
                base_url: std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
                model: std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "text-embedding-3-small".to_string()),
                dimension: std::env::var("EMBEDDING_DIMENSION").unwrap_or_else(|_| "1536".to_string()).parse().unwrap_or(1536),
                headers: HashMap::new(),
            },
            EmbeddingProvider::Ollama => EmbeddingConfig {
                provider_type: "ollama".to_string(),
                api_key: String::new(),
                base_url: std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string()),
                model: std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "all-minilm".to_string()),
                dimension: std::env::var("EMBEDDING_DIMENSION").unwrap_or_else(|_| "384".to_string()).parse().unwrap_or(384),
                headers: HashMap::new(),
            },
            EmbeddingProvider::HuggingFace => EmbeddingConfig {
                provider_type: "huggingface".to_string(),
                api_key: String::new(),
                base_url: String::new(),
                model: std::env::var("HUGGINGFACE_MODEL").unwrap_or_else(|_| "sentence-transformers/all-MiniLM-L6-v2".to_string()),
                dimension: std::env::var("EMBEDDING_DIMENSION").unwrap_or_else(|_| "384".to_string()).parse().unwrap_or(384),
                headers: HashMap::new(),
            },
            EmbeddingProvider::Custom => {
                let mut headers = HashMap::new();
                // Load any CUSTOM_EMBEDDING_HEADER_* env vars
                for (key, value) in std::env::vars() {
                    if key.starts_with("CUSTOM_EMBEDDING_HEADER_") {
                        let header_name = key.strip_prefix("CUSTOM_EMBEDDING_HEADER_").unwrap().replace('_', "-").to_lowercase();
                        headers.insert(header_name, value);
                    }
                }
                
                EmbeddingConfig {
                    provider_type: "custom".to_string(),
                    api_key: std::env::var("CUSTOM_EMBEDDING_API_KEY").unwrap_or_default(),
                    base_url: std::env::var("CUSTOM_EMBEDDING_URL").unwrap_or_default(),
                    model: std::env::var("CUSTOM_EMBEDDING_MODEL").unwrap_or_else(|_| "default".to_string()),
                    dimension: std::env::var("EMBEDDING_DIMENSION").unwrap_or_else(|_| "1536".to_string()).parse().unwrap_or(1536),
                    headers,
                }
            }
        }
    }
    
    /// Get storage configuration details
    pub fn storage_config(&self) -> StorageConfig {
        match &self.storage {
            StorageBackend::SQLiteInMemory => StorageConfig {
                metadata_type: "sqlite".to_string(),
                metadata_url: std::env::var("SQLITE_PATH").unwrap_or_else(|_| "./memory.db".to_string()),
                vector_type: "memory".to_string(),
                vector_url: String::new(),
                vector_api_key: None,
                collection_name: "memory".to_string(),
            },
            StorageBackend::SQLiteQdrant => StorageConfig {
                metadata_type: "sqlite".to_string(),
                metadata_url: std::env::var("SQLITE_PATH").unwrap_or_else(|_| "./memory.db".to_string()),
                vector_type: "qdrant".to_string(),
                vector_url: std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".to_string()),
                vector_api_key: std::env::var("QDRANT_API_KEY").ok(),
                collection_name: std::env::var("QDRANT_COLLECTION").unwrap_or_else(|_| "memory".to_string()),
            },
            StorageBackend::PostgreSQLInMemory => StorageConfig {
                metadata_type: "postgresql".to_string(),
                metadata_url: std::env::var("DATABASE_URL").unwrap_or_default(),
                vector_type: "memory".to_string(),
                vector_url: String::new(),
                vector_api_key: None,
                collection_name: "memory".to_string(),
            },
            StorageBackend::PostgreSQLQdrant => StorageConfig {
                metadata_type: "postgresql".to_string(),
                metadata_url: std::env::var("DATABASE_URL").unwrap_or_default(),
                vector_type: "qdrant".to_string(),
                vector_url: std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".to_string()),
                vector_api_key: std::env::var("QDRANT_API_KEY").ok(),
                collection_name: std::env::var("QDRANT_COLLECTION").unwrap_or_else(|_| "memory".to_string()),
            },
            StorageBackend::PostgreSQLPgVector => StorageConfig {
                metadata_type: "postgresql".to_string(),
                metadata_url: std::env::var("DATABASE_URL").unwrap_or_default(),
                vector_type: "pgvector".to_string(),
                vector_url: std::env::var("DATABASE_URL").unwrap_or_default(),
                vector_api_key: None,
                collection_name: "embeddings".to_string(),
            },
            StorageBackend::MySQLInMemory => StorageConfig {
                metadata_type: "mysql".to_string(),
                metadata_url: std::env::var("DATABASE_URL").unwrap_or_default(),
                vector_type: "memory".to_string(),
                vector_url: String::new(),
                vector_api_key: None,
                collection_name: "memory".to_string(),
            },
            StorageBackend::MySQLQdrant => StorageConfig {
                metadata_type: "mysql".to_string(),
                metadata_url: std::env::var("DATABASE_URL").unwrap_or_default(),
                vector_type: "qdrant".to_string(),
                vector_url: std::env::var("QDRANT_URL").unwrap_or_else(|_| "http://localhost:6333".to_string()),
                vector_api_key: std::env::var("QDRANT_API_KEY").ok(),
                collection_name: std::env::var("QDRANT_COLLECTION").unwrap_or_else(|_| "memory".to_string()),
            },
        }
    }
}

/// Internal embedding configuration
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    pub provider_type: String,
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub dimension: usize,
    pub headers: HashMap<String, String>,
}

/// Internal storage configuration
#[derive(Debug, Clone)]
pub struct StorageConfig {
    pub metadata_type: String,
    pub metadata_url: String,
    pub vector_type: String,
    pub vector_url: String,
    pub vector_api_key: Option<String>,
    pub collection_name: String,
} 