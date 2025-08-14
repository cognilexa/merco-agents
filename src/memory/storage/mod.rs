use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use super::{MemoryEntry, MemoryType, MemoryQuery, MemoryResult};
use super::config::{StorageConfig};

/// Storage backend error types
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("Database error: {0}")]
    DatabaseError(String),
    #[error("Connection failed: {0}")]
    ConnectionError(String),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("Vector store error: {0}")]
    VectorError(String),
}

/// Persistent metadata storage trait
#[async_trait]
pub trait MetadataStorage: Send + Sync {
    async fn store_metadata(&mut self, entry: &MemoryEntry) -> Result<(), StorageError>;
    async fn get_metadata(&self, id: &str) -> Result<Option<MemoryEntry>, StorageError>;
    async fn update_metadata(&mut self, id: &str, entry: &MemoryEntry) -> Result<(), StorageError>;
    async fn delete_metadata(&mut self, id: &str) -> Result<(), StorageError>;
    async fn list_by_type(&self, memory_type: MemoryType, limit: usize) -> Result<Vec<MemoryEntry>, StorageError>;
    async fn list_by_user(&self, user_id: &str, limit: usize) -> Result<Vec<MemoryEntry>, StorageError>;
    async fn search_metadata(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>, StorageError>;
}

/// Vector storage trait for embeddings
#[async_trait]
pub trait VectorStorage: Send + Sync {
    async fn store_vector(&mut self, id: &str, vector: &[f32], metadata: HashMap<String, String>) -> Result<(), StorageError>;
    async fn search_vectors(&self, query_vector: &[f32], limit: usize, similarity_threshold: f32) -> Result<Vec<VectorSearchResult>, StorageError>;
    async fn delete_vector(&mut self, id: &str) -> Result<(), StorageError>;
    async fn get_vector(&self, id: &str) -> Result<Option<Vec<f32>>, StorageError>;
}

#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    pub id: String,
    pub score: f32,
    pub metadata: HashMap<String, String>,
}

/// SQLite metadata storage implementation
pub struct SqliteMetadataStorage {
    pool: sqlx::sqlite::SqlitePool,
}

impl SqliteMetadataStorage {
    pub async fn new(database_path: &str) -> Result<Self, StorageError> {
        // Create database connection with create_if_missing option
        use sqlx::sqlite::SqliteConnectOptions;
        use std::str::FromStr;
        
        let options = SqliteConnectOptions::from_str(&format!("sqlite:{}", database_path))
            .map_err(|e| StorageError::ConfigError(e.to_string()))?
            .create_if_missing(true);
        
        let pool = sqlx::sqlite::SqlitePool::connect_with(options)
            .await
            .map_err(|e| StorageError::ConnectionError(e.to_string()))?;

        // Create tables
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS memory_entries (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                memory_type TEXT NOT NULL,
                relevance_score REAL,
                user_id TEXT,
                agent_id TEXT
            )
            "#,
        )
        .execute(&pool)
        .await
        .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        // Create indexes
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_type ON memory_entries(memory_type)")
            .execute(&pool)
            .await
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_user_id ON memory_entries(user_id)")
            .execute(&pool)
            .await
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_timestamp ON memory_entries(timestamp)")
            .execute(&pool)
            .await
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        Ok(Self { pool })
    }
}

#[async_trait]
impl MetadataStorage for SqliteMetadataStorage {
    async fn store_metadata(&mut self, entry: &MemoryEntry) -> Result<(), StorageError> {
        let metadata_json = serde_json::to_string(&entry.metadata)?;
        let memory_type_str = format!("{:?}", entry.memory_type);
        
        let user_id = entry.metadata.get("user_id").cloned();
        let agent_id = entry.metadata.get("agent_id").cloned();

        sqlx::query(
            r#"
            INSERT OR REPLACE INTO memory_entries 
            (id, content, metadata, timestamp, memory_type, relevance_score, user_id, agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&entry.id)
        .bind(&entry.content)
        .bind(metadata_json)
        .bind(entry.timestamp)
        .bind(memory_type_str)
        .bind(entry.relevance_score)
        .bind(user_id)
        .bind(agent_id)
        .execute(&self.pool)
        .await
        .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        Ok(())
    }

    async fn get_metadata(&self, id: &str) -> Result<Option<MemoryEntry>, StorageError> {
        let row = sqlx::query_as::<_, (String, String, String, DateTime<Utc>, String, Option<f32>)>(
            "SELECT id, content, metadata, timestamp, memory_type, relevance_score FROM memory_entries WHERE id = ?"
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        if let Some((id, content, metadata_json, timestamp, memory_type_str, relevance_score)) = row {
            let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json)?;
            let memory_type = match memory_type_str.as_str() {
                "Working" => MemoryType::Working,
                "Semantic" => MemoryType::Semantic,
                "Procedural" => MemoryType::Procedural,
                "Episodic" => MemoryType::Episodic,
                _ => MemoryType::Semantic, // Default fallback
            };

            Ok(Some(MemoryEntry {
                id,
                content,
                metadata,
                timestamp,
                memory_type,
                relevance_score,
                embeddings: None, // Vector data stored separately
            }))
        } else {
            Ok(None)
        }
    }

    async fn update_metadata(&mut self, id: &str, entry: &MemoryEntry) -> Result<(), StorageError> {
        self.store_metadata(entry).await
    }

    async fn delete_metadata(&mut self, id: &str) -> Result<(), StorageError> {
        sqlx::query("DELETE FROM memory_entries WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(|e| StorageError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    async fn list_by_type(&self, memory_type: MemoryType, limit: usize) -> Result<Vec<MemoryEntry>, StorageError> {
        let memory_type_str = format!("{:?}", memory_type);
        let rows = sqlx::query_as::<_, (String, String, String, DateTime<Utc>, String, Option<f32>)>(
            "SELECT id, content, metadata, timestamp, memory_type, relevance_score 
             FROM memory_entries WHERE memory_type = ? 
             ORDER BY timestamp DESC LIMIT ?"
        )
        .bind(memory_type_str)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        let mut entries = Vec::new();
        for (id, content, metadata_json, timestamp, _, relevance_score) in rows {
            let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json)?;
            entries.push(MemoryEntry {
                id,
                content,
                metadata,
                timestamp,
                memory_type: memory_type.clone(),
                relevance_score,
                embeddings: None,
            });
        }

        Ok(entries)
    }

    async fn list_by_user(&self, user_id: &str, limit: usize) -> Result<Vec<MemoryEntry>, StorageError> {
        let rows = sqlx::query_as::<_, (String, String, String, DateTime<Utc>, String, Option<f32>)>(
            "SELECT id, content, metadata, timestamp, memory_type, relevance_score 
             FROM memory_entries WHERE user_id = ? 
             ORDER BY timestamp DESC LIMIT ?"
        )
        .bind(user_id)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        let mut entries = Vec::new();
        for (id, content, metadata_json, timestamp, memory_type_str, relevance_score) in rows {
            let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json)?;
            let memory_type = match memory_type_str.as_str() {
                "Working" => MemoryType::Working,
                "Semantic" => MemoryType::Semantic,
                "Procedural" => MemoryType::Procedural,
                "Episodic" => MemoryType::Episodic,
                _ => MemoryType::Semantic,
            };

            entries.push(MemoryEntry {
                id,
                content,
                metadata,
                timestamp,
                memory_type,
                relevance_score,
                embeddings: None,
            });
        }

        Ok(entries)
    }

    async fn search_metadata(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>, StorageError> {
        let search_term = format!("%{}%", query);
        let rows = sqlx::query_as::<_, (String, String, String, DateTime<Utc>, String, Option<f32>)>(
            "SELECT id, content, metadata, timestamp, memory_type, relevance_score 
             FROM memory_entries WHERE content LIKE ? 
             ORDER BY timestamp DESC LIMIT ?"
        )
        .bind(search_term)
        .bind(limit as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| StorageError::DatabaseError(e.to_string()))?;

        let mut entries = Vec::new();
        for (id, content, metadata_json, timestamp, memory_type_str, relevance_score) in rows {
            let metadata: HashMap<String, String> = serde_json::from_str(&metadata_json)?;
            let memory_type = match memory_type_str.as_str() {
                "Working" => MemoryType::Working,
                "Semantic" => MemoryType::Semantic,
                "Procedural" => MemoryType::Procedural,
                "Episodic" => MemoryType::Episodic,
                _ => MemoryType::Semantic,
            };

            entries.push(MemoryEntry {
                id,
                content,
                metadata,
                timestamp,
                memory_type,
                relevance_score,
                embeddings: None,
            });
        }

        Ok(entries)
    }
}

/// In-memory vector storage (for development/testing)
pub struct InMemoryVectorStorage {
    vectors: HashMap<String, (Vec<f32>, HashMap<String, String>)>,
}

impl InMemoryVectorStorage {
    pub fn new() -> Self {
        Self {
            vectors: HashMap::new(),
        }
    }
    
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

#[async_trait]
impl VectorStorage for InMemoryVectorStorage {
    async fn store_vector(&mut self, id: &str, vector: &[f32], metadata: HashMap<String, String>) -> Result<(), StorageError> {
        self.vectors.insert(id.to_string(), (vector.to_vec(), metadata));
        Ok(())
    }

    async fn search_vectors(&self, query_vector: &[f32], limit: usize, similarity_threshold: f32) -> Result<Vec<VectorSearchResult>, StorageError> {
        let mut results: Vec<VectorSearchResult> = self.vectors
            .iter()
            .map(|(id, (vector, metadata))| {
                let score = Self::cosine_similarity(query_vector, vector);
                VectorSearchResult {
                    id: id.clone(),
                    score,
                    metadata: metadata.clone(),
                }
            })
            .filter(|result| result.score >= similarity_threshold)
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        
        Ok(results)
    }

    async fn delete_vector(&mut self, id: &str) -> Result<(), StorageError> {
        self.vectors.remove(id);
        Ok(())
    }

    async fn get_vector(&self, id: &str) -> Result<Option<Vec<f32>>, StorageError> {
        Ok(self.vectors.get(id).map(|(vector, _)| vector.clone()))
    }
}

/// Qdrant vector storage implementation
pub struct QdrantVectorStorage {
    client: qdrant_client::Qdrant,
    collection_name: String,
}

impl QdrantVectorStorage {
    pub async fn new(url: &str, api_key: Option<String>, collection_name: String) -> Result<Self, StorageError> {
        let mut client_builder = qdrant_client::Qdrant::from_url(url);
        
        if let Some(key) = api_key {
            client_builder = client_builder.api_key(key);
        }
        
        let client = client_builder.build()
            .map_err(|e| StorageError::ConnectionError(e.to_string()))?;
            
        Ok(Self {
            client,
            collection_name,
        })
    }
}

#[async_trait]
impl VectorStorage for QdrantVectorStorage {
    async fn store_vector(&mut self, id: &str, vector: &[f32], metadata: HashMap<String, String>) -> Result<(), StorageError> {
        use qdrant_client::qdrant::{PointStruct, UpsertPointsBuilder};
        
        let mut payload = qdrant_client::Payload::new();
        for (k, v) in metadata {
            payload.insert(k, v);
        }

        let point = PointStruct::new(id, vector.to_vec(), payload);

        self.client
            .upsert_points(
                UpsertPointsBuilder::new(&self.collection_name, vec![point])
                    .wait(true)
            )
            .await
            .map_err(|e| StorageError::VectorError(e.to_string()))?;

        Ok(())
    }

    async fn search_vectors(&self, query_vector: &[f32], limit: usize, similarity_threshold: f32) -> Result<Vec<VectorSearchResult>, StorageError> {
        use qdrant_client::qdrant::SearchPointsBuilder;

        let search_result = self.client
            .search_points(
                SearchPointsBuilder::new(&self.collection_name, query_vector.to_vec(), limit as u64)
                    .score_threshold(similarity_threshold)
                    .with_payload(true)
            )
            .await
            .map_err(|e| StorageError::VectorError(e.to_string()))?;

        let results: Vec<VectorSearchResult> = search_result
            .result
            .into_iter()
            .map(|point| {
                let id = match point.id {
                    Some(point_id) => {
                        match point_id.point_id_options {
                            Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid)) => uuid,
                            Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) => num.to_string(),
                            None => "unknown".to_string(),
                        }
                    }
                    None => "unknown".to_string(),
                };

                let metadata: HashMap<String, String> = point.payload
                    .into_iter()
                    .filter_map(|(k, v)| {
                        // Convert Qdrant Value to String - simplified conversion
                        match v.kind {
                            Some(qdrant_client::qdrant::value::Kind::StringValue(s)) => Some((k, s)),
                            Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)) => Some((k, i.to_string())),
                            Some(qdrant_client::qdrant::value::Kind::DoubleValue(d)) => Some((k, d.to_string())),
                            Some(qdrant_client::qdrant::value::Kind::BoolValue(b)) => Some((k, b.to_string())),
                            _ => None,
                        }
                    })
                    .collect();

                VectorSearchResult {
                    id,
                    score: point.score,
                    metadata,
                }
            })
            .collect();

        Ok(results)
    }

    async fn delete_vector(&mut self, id: &str) -> Result<(), StorageError> {
        use qdrant_client::qdrant::{DeletePointsBuilder, PointsIdsList};

        let point_id = qdrant_client::qdrant::PointId {
            point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(id.to_string()))
        };

        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection_name)
                    .points(PointsIdsList {
                        ids: vec![point_id],
                    })
                    .wait(true)
            )
            .await
            .map_err(|e| StorageError::VectorError(e.to_string()))?;

        Ok(())
    }

    async fn get_vector(&self, id: &str) -> Result<Option<Vec<f32>>, StorageError> {
        // Qdrant doesn't have a direct get_vector method, would need to use retrieve_points
        // For now, return None (vectors are typically retrieved through search)
        Ok(None)
    }
}

/// Factory functions for creating storage backends
pub async fn create_metadata_storage(config: &StorageConfig) -> Result<Box<dyn MetadataStorage>, StorageError> {
    match config.metadata_type.as_str() {
        "sqlite" => {
            let storage = SqliteMetadataStorage::new(&config.metadata_url).await?;
            Ok(Box::new(storage))
        }
        "postgresql" => {
            // TODO: Implement PostgreSQL storage
            Err(StorageError::ConfigError("PostgreSQL not yet implemented".to_string()))
        }
        "mysql" => {
            // TODO: Implement MySQL storage
            Err(StorageError::ConfigError("MySQL not yet implemented".to_string()))
        }
        _ => Err(StorageError::ConfigError(format!("Unknown metadata storage type: {}", config.metadata_type)))
    }
}

pub async fn create_vector_storage(config: &StorageConfig) -> Result<Box<dyn VectorStorage>, StorageError> {
    match config.vector_type.as_str() {
        "memory" => {
            Ok(Box::new(InMemoryVectorStorage::new()))
        }
        "qdrant" => {
            let storage = QdrantVectorStorage::new(&config.vector_url, config.vector_api_key.clone(), config.collection_name.clone()).await?;
            Ok(Box::new(storage))
        }
        "pgvector" => {
            Err(StorageError::ConfigError("PostgreSQL pgvector not yet implemented".to_string()))
        }
        _ => {
            Err(StorageError::ConfigError(format!("Unknown vector storage type: {}", config.vector_type)))
        }
    }
} 