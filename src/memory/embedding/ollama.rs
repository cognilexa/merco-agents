use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use super::{EmbeddingProviderTrait, EmbeddingError};

/// Ollama embedding provider
pub struct OllamaEmbeddingProvider {
    client: reqwest::Client,
    base_url: String,
    model: String,
    dimension: usize,
}

impl OllamaEmbeddingProvider {
    pub fn new(base_url: String, model: String, dimension: usize) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url,
            model,
            dimension,
        }
    }
}

#[async_trait]
impl EmbeddingProviderTrait for OllamaEmbeddingProvider {
    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut embeddings = Vec::new();
        
        for text in texts {
            #[derive(Serialize)]
            struct OllamaEmbeddingRequest {
                model: String,
                prompt: String,
            }

            #[derive(Deserialize)]
            struct OllamaEmbeddingResponse {
                embedding: Vec<f32>,
            }

            let request = OllamaEmbeddingRequest {
                model: self.model.clone(),
                prompt: text.clone(),
            };

            let response = self.client
                .post(&format!("{}/api/embeddings", self.base_url))
                .json(&request)
                .send()
                .await?;

            if !response.status().is_success() {
                let error_text = response.text().await.unwrap_or_default();
                return Err(EmbeddingError::ApiError { message: error_text });
            }

            let embedding_response: OllamaEmbeddingResponse = response.json().await?;
            embeddings.push(embedding_response.embedding);
        }

        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
} 