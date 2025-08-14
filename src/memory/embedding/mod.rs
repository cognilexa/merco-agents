mod types;
mod openai;
mod ollama;
mod huggingface;
mod custom;

pub use types::*;
pub use openai::OpenAIEmbeddingProvider;
pub use ollama::OllamaEmbeddingProvider;
pub use huggingface::HuggingFaceEmbeddingProvider;
pub use custom::CustomEmbeddingProvider;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use super::config::{EmbeddingConfig};

// Types that were in config but now need to be here
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HuggingFaceDevice {
    Cpu,
    Cuda(usize), // GPU index
    Metal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequestFormat {
    OpenAICompatible,
    Custom {
        text_field: String,
        response_field: String,
    },
}

/// Embedding provider trait
#[async_trait]
pub trait EmbeddingProviderTrait: Send + Sync {
    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError>;
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let results = self.embed_texts(&[text.to_string()]).await?;
        results.into_iter().next().ok_or(EmbeddingError::EmptyResponse)
    }
    fn dimension(&self) -> usize;
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),
    #[error("JSON parsing failed: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("API error: {message}")]
    ApiError { message: String },
    #[error("Empty response from embedding provider")]
    EmptyResponse,
    #[error("Model loading failed: {0}")]
    ModelError(String),
    #[error("Invalid configuration: {0}")]
    ConfigError(String),
}

/// Factory function to create embedding providers
pub fn create_embedding_provider(config: &EmbeddingConfig) -> Result<Box<dyn EmbeddingProviderTrait>, EmbeddingError> {
    match config.provider_type.as_str() {
        "openai" => {
            Ok(Box::new(OpenAIEmbeddingProvider::new(
                config.api_key.clone(),
                config.model.clone(),
                Some(config.base_url.clone()),
                config.dimension,
            )))
        }
        "ollama" => {
            Ok(Box::new(OllamaEmbeddingProvider::new(
                config.base_url.clone(),
                config.model.clone(),
                config.dimension,
            )))
        }
        "huggingface" => {
            Ok(Box::new(HuggingFaceEmbeddingProvider::new(
                config.model.clone(),
                None,
                HuggingFaceDevice::Cpu,
                config.dimension,
            )))
        }
        "custom" => {
            Ok(Box::new(CustomEmbeddingProvider::new(
                config.base_url.clone(),
                config.headers.clone(),
                RequestFormat::OpenAICompatible,
                config.dimension,
            )))
        }
        _ => Err(EmbeddingError::ConfigError(format!(
            "Unknown embedding provider type: {}",
            config.provider_type
        )))
    }
} 