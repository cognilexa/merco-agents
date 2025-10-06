use serde::{Deserialize, Serialize};

/// LLM Provider types supported by merco-agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Provider {
    /// OpenAI API
    OpenAI,
    /// Anthropic Claude API
    Anthropic,
    /// Google Gemini API
    Google,
    /// Ollama local models
    Ollama,
    /// Custom provider with custom base URL
    Custom(String),
}

impl Provider {
    /// Convert our Provider to merco_llmproxy Provider
    pub fn to_llmproxy_provider(&self) -> merco_llmproxy::config::Provider {
        match self {
            Provider::OpenAI => merco_llmproxy::config::Provider::OpenAI,
            Provider::Anthropic => merco_llmproxy::config::Provider::Anthropic,
            Provider::Google => merco_llmproxy::config::Provider::OpenAI, // Map Google to OpenAI for now
            Provider::Ollama => merco_llmproxy::config::Provider::Ollama,
            Provider::Custom(_) => merco_llmproxy::config::Provider::Custom,
        }
    }

    /// Get the base URL for the provider
    pub fn get_base_url(&self) -> Option<String> {
        match self {
            Provider::OpenAI => Some("https://api.openai.com/v1".to_string()),
            Provider::Anthropic => Some("https://api.anthropic.com".to_string()),
            Provider::Google => Some("https://generativelanguage.googleapis.com/v1beta".to_string()),
            Provider::Ollama => Some("http://localhost:11434".to_string()),
            Provider::Custom(url) => Some(url.clone()),
        }
    }
}

/// LLM Configuration for merco-agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    /// The provider to use
    pub provider: Provider,
    /// API key for the provider
    pub api_key: Option<String>,
    /// Custom base URL (overrides default for provider)
    pub base_url: Option<String>,
    /// Additional headers for the request
    pub headers: Option<std::collections::HashMap<String, String>>,
}

impl LlmConfig {
    /// Create a new LLM configuration
    pub fn new(provider: Provider, api_key: Option<String>) -> Self {
        Self {
            provider,
            api_key,
            base_url: None,
            headers: None,
        }
    }

    /// Create a new LLM configuration with custom base URL
    pub fn new_with_base_url(provider: Provider, api_key: Option<String>, base_url: String) -> Self {
        Self {
            provider,
            api_key,
            base_url: Some(base_url),
            headers: None,
        }
    }

    /// Convert to merco_llmproxy LlmConfig
    pub fn to_llmproxy_config(&self) -> merco_llmproxy::LlmConfig {
        merco_llmproxy::LlmConfig {
            provider: self.provider.to_llmproxy_provider(),
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone().or_else(|| self.provider.get_base_url()),
        }
    }
}
