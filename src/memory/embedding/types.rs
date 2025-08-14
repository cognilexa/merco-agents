use serde::{Deserialize, Serialize};

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