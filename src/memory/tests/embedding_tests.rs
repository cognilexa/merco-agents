use super::super::embedding::*;
use super::super::config::*;
use std::env;

#[tokio::test]
async fn test_openai_embedding_provider() {
    // Skip if no API key
    if env::var("OPENAI_API_KEY").is_err() {
        return;
    }

    let config = EmbeddingConfig {
        provider_type: "openai".to_string(),
        api_key: env::var("OPENAI_API_KEY").unwrap(),
        base_url: "https://api.openai.com/v1".to_string(),
        model: "text-embedding-3-small".to_string(),
        dimension: 1536,
        headers: std::collections::HashMap::new(),
    };

    let provider = create_embedding_provider(&config).unwrap();
    let result = provider.embed_text("Test text").await;
    
    assert!(result.is_ok());
    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 1536);
}

#[tokio::test]
async fn test_ollama_embedding_provider() {
    // Skip if Ollama not available
    if env::var("OLLAMA_URL").is_err() {
        return;
    }

    let config = EmbeddingConfig {
        provider_type: "ollama".to_string(),
        api_key: String::new(),
        base_url: env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string()),
        model: "all-minilm".to_string(),
        dimension: 384,
        headers: std::collections::HashMap::new(),
    };

    let provider = create_embedding_provider(&config).unwrap();
    let result = provider.embed_text("Test text").await;
    
    assert!(result.is_ok());
    let embeddings = result.unwrap();
    assert_eq!(embeddings.len(), 384);
}

#[test]
fn test_invalid_provider_type() {
    let config = EmbeddingConfig {
        provider_type: "invalid".to_string(),
        api_key: String::new(),
        base_url: String::new(),
        model: String::new(),
        dimension: 0,
        headers: std::collections::HashMap::new(),
    };

    let result = create_embedding_provider(&config);
    assert!(result.is_err());
}

#[test]
fn test_missing_api_key() {
    env::remove_var("OPENAI_API_KEY");
    
    let config = EmbeddingConfig {
        provider_type: "openai".to_string(),
        api_key: String::new(),
        base_url: "https://api.openai.com/v1".to_string(),
        model: "text-embedding-3-small".to_string(),
        dimension: 1536,
        headers: std::collections::HashMap::new(),
    };

    let result = create_embedding_provider(&config);
    assert!(result.is_err());
} 