use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Defines the role and specialization of an agent
/// Completely flexible - users define their own roles
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentRole {
    /// Name of the role (e.g., "Researcher", "Writer", "Analyst", "CustomRole")
    pub name: String,
    /// Description of what this role does
    pub description: String,
    /// Custom metadata for role-specific configuration
    pub metadata: HashMap<String, serde_json::Value>,
}


impl AgentRole {
    /// Create a new custom role
    pub fn new(
        name: String,
        description: String,
    ) -> Self {
        Self {
            name,
            description,
            metadata: HashMap::new(),
        }
    }

    /// Add custom metadata to the role
    pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get a human-readable description of the role
    pub fn get_description(&self) -> String {
        format!("{}: {}", self.name, self.description)
    }

    /// Get metadata value by key
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }

    /// Set metadata value
    pub fn set_metadata(&mut self, key: String, value: serde_json::Value) {
        self.metadata.insert(key, value);
    }
}

/// Agent capabilities and limitations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapabilities {
    pub max_concurrent_tasks: usize,
    pub supported_output_formats: Vec<OutputFormat>,
}

// InputType simplified to just Text for now
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InputType {
    Text,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OutputFormat {
    Text,
    Json,
    Markdown,
    Html,
    MultiModal,
}


