use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use super::{EmbeddingProviderTrait, EmbeddingError};

/// OpenAI embedding provider
pub struct OpenAIEmbeddingProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    dimension: usize,
}

impl OpenAIEmbeddingProvider {
    pub fn new(api_key: String, model: String, base_url: Option<String>, dimension: usize) -> Self {
        let base_url = base_url.unwrap_or_else(|| "https://api.openai.com/v1".to_string());
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            base_url,
            dimension,
        }
    }
}

#[async_trait]
impl EmbeddingProviderTrait for OpenAIEmbeddingProvider {
    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        #[derive(Serialize)]
        struct EmbeddingRequest {
            input: Vec<String>,
            model: String,
        }

        #[derive(Deserialize)]
        struct EmbeddingResponse {
            data: Vec<EmbeddingData>,
        }

        #[derive(Deserialize)]
        struct EmbeddingData {
            embedding: Vec<f32>,
        }

        let request = EmbeddingRequest {
            input: texts.to_vec(),
            model: self.model.clone(),
        };

        let response = self.client
            .post(&format!("{}/embeddings", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(EmbeddingError::ApiError { message: error_text });
        }

        let embedding_response: EmbeddingResponse = response.json().await?;
        Ok(embedding_response.data.into_iter().map(|d| d.embedding).collect())
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
} 