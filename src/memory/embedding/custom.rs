use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use super::{EmbeddingProviderTrait, EmbeddingError, RequestFormat};

/// Custom embedding provider for any URL-based service
/// OpenAI compatible API is supported by default
pub struct CustomEmbeddingProvider {
    client: reqwest::Client,
    url: String,
    headers: HashMap<String, String>,
    request_format: RequestFormat,
    dimension: usize,
}

impl CustomEmbeddingProvider {
    pub fn new(url: String, headers: HashMap<String, String>, request_format: RequestFormat, dimension: usize) -> Self {
        Self {
            client: reqwest::Client::new(),
            url,
            headers,
            request_format,
            dimension,
        }
    }
}

#[async_trait]
impl EmbeddingProviderTrait for CustomEmbeddingProvider {
    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        match &self.request_format {
            RequestFormat::OpenAICompatible => {
                #[derive(Serialize)]
                struct Request {
                    input: Vec<String>,
                    model: String,
                }

                #[derive(Deserialize)]
                struct Response {
                    data: Vec<EmbeddingData>,
                }

                #[derive(Deserialize)]
                struct EmbeddingData {
                    embedding: Vec<f32>,
                }

                let request = Request {
                    input: texts.to_vec(),
                    model: "default".to_string(),
                };

                let mut req = self.client.post(&self.url).json(&request);
                for (key, value) in &self.headers {
                    req = req.header(key, value);
                }

                let response = req.send().await?;
                if !response.status().is_success() {
                    let error_text = response.text().await.unwrap_or_default();
                    return Err(EmbeddingError::ApiError { message: error_text });
                }

                let embedding_response: Response = response.json().await?;
                Ok(embedding_response.data.into_iter().map(|d| d.embedding).collect())
            }
            RequestFormat::Custom { text_field, response_field } => {
                let mut embeddings = Vec::new();
                for text in texts {
                    let mut request_body = serde_json::Map::new();
                    request_body.insert(text_field.clone(), serde_json::Value::String(text.clone()));

                    let mut req = self.client.post(&self.url).json(&request_body);
                    for (key, value) in &self.headers {
                        req = req.header(key, value);
                    }

                    let response = req.send().await?;
                    if !response.status().is_success() {
                        let error_text = response.text().await.unwrap_or_default();
                        return Err(EmbeddingError::ApiError { message: error_text });
                    }

                    let response_json: serde_json::Value = response.json().await?;
                    if let Some(embedding_value) = response_json.get(response_field) {
                        if let Some(embedding_array) = embedding_value.as_array() {
                            let embedding: Result<Vec<f32>, _> = embedding_array
                                .iter()
                                .map(|v| v.as_f64().map(|f| f as f32).ok_or(EmbeddingError::EmptyResponse))
                                .collect();
                            embeddings.push(embedding?);
                        }
                    }
                }
                Ok(embeddings)
            }
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
} 