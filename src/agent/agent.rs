use crate::agent::role::{AgentRole, AgentCapabilities};
use crate::agent::state::{AgentState, AgentContext};
use crate::agent::output_handler::OutputHandler;
use crate::agent::provider::LlmConfig;
use merco_llmproxy::{LlmProvider, Tool};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use chrono;

/// Core Agent structure
#[derive(Clone)]
pub struct Agent {
    // Basic Information
    pub id: String,
    pub name: String,
    pub description: String,
    
    // Role and Capabilities
    pub role: AgentRole,
    pub capabilities: AgentCapabilities,
    
    // LLM Configuration
    pub llm_config: AgentModelConfig,
    
    // Tools
    pub tools: Vec<Tool>,
    
    // State and Context
    pub state: AgentState,
    pub context: AgentContext,
    
    // Output handling
    pub output_handler: OutputHandler,
    
    // LLM Provider
    pub provider: Arc<dyn LlmProvider + Send + Sync>,
}

/// LLM Configuration for agents
#[derive(Debug, Clone)]
pub struct AgentModelConfig {
    pub model_name: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub llm_config: LlmConfig,
}

impl AgentModelConfig {
    pub fn new(llm_config: LlmConfig, model_name: String, temperature: f32, max_tokens: u32) -> Self {
        Self {
            model_name,
            temperature,
            max_tokens,
            llm_config,
        }
    }

    /// Convert to merco_llmproxy LlmConfig
    pub fn to_llmproxy_config(&self) -> merco_llmproxy::LlmConfig {
        self.llm_config.to_llmproxy_config()
    }
}

/// Task execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub success: bool,
    pub output: String,
    pub execution_time_ms: u64,
    pub tokens_used: u32,
    pub tools_used: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Agent error types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentError {
    AgentBusy,
    InvalidTask,
    LLMError(String),
    ToolError(String),
    ValidationError(String),
    TooManyConcurrentTasks,
    AgentNotFound,
    InvalidConfiguration,
}

impl std::fmt::Display for AgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentError::AgentBusy => write!(f, "Agent is currently busy"),
            AgentError::InvalidTask => write!(f, "Invalid task provided"),
            AgentError::LLMError(msg) => write!(f, "LLM error: {}", msg),
            AgentError::ToolError(msg) => write!(f, "Tool error: {}", msg),
            AgentError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            AgentError::TooManyConcurrentTasks => write!(f, "Too many concurrent tasks"),
            AgentError::AgentNotFound => write!(f, "Agent not found"),
            AgentError::InvalidConfiguration => write!(f, "Invalid configuration"),
        }
    }
}

impl std::error::Error for AgentError {}

/// Detailed information about a tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Name of the tool that was called
    pub tool_name: String,
    /// Parameters passed to the tool (as JSON string)
    pub parameters: String,
    /// Result returned by the tool (as JSON string)
    pub result: String,
    /// Time taken to execute the tool in milliseconds
    pub execution_time_ms: u64,
    /// Any error that occurred during tool execution
    pub error: Option<String>,
    /// Output format of the tool result
    pub output_format: String,
}

impl ToolCall {
    pub fn new(
        tool_name: String,
        parameters: String,
        result: String,
        execution_time_ms: u64,
        output_format: String,
    ) -> Self {
        Self {
            tool_name,
            parameters,
            result,
            execution_time_ms,
            error: None,
            output_format,
        }
    }

    pub fn with_error(
        tool_name: String,
        parameters: String,
        error: String,
        execution_time_ms: u64,
        output_format: String,
    ) -> Self {
        Self {
            tool_name,
            parameters,
            result: String::new(),
            execution_time_ms,
            error: Some(error),
            output_format,
        }
    }
}

// Agent Response structure with comprehensive metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResponse {
    /// The actual response content from the agent
    pub content: String,
    /// Whether the task was completed successfully
    pub success: bool,
    /// Time taken to complete the task in milliseconds
    pub execution_time_ms: u64,
    /// Number of tokens used in the request
    pub input_tokens: u32,
    /// Number of tokens generated in the response
    pub output_tokens: u32,
    /// Total tokens used (input + output)
    pub total_tokens: u32,
    /// Tools that were used during execution
    pub tools_used: Vec<String>,
    /// Detailed information about tool calls made during execution
    pub tool_calls: Vec<ToolCall>,
    /// Number of tool calls made
    pub tool_calls_count: usize,
    /// Total time spent executing tools in milliseconds
    pub tool_execution_time_ms: u64,
    /// Output format of the agent's response
    pub output_format: String,
    /// Model used for the response
    pub model_used: String,
    /// Temperature setting used
    pub temperature: f32,
    /// Any error message if the task failed
    pub error: Option<String>,
    /// Additional metadata about the execution
    pub metadata: HashMap<String, serde_json::Value>,
    /// Timestamp when the response was generated
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl AgentResponse {
    /// Create a successful response
    pub fn success(
        content: String,
        execution_time_ms: u64,
        input_tokens: u32,
        output_tokens: u32,
        model_used: String,
        temperature: f32,
        tools_used: Vec<String>,
        tool_calls: Vec<ToolCall>,
        output_format: String,
    ) -> Self {
        let tool_execution_time_ms = tool_calls.iter().map(|tc| tc.execution_time_ms).sum();
        Self {
            content,
            success: true,
            execution_time_ms,
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
            tools_used,
            tool_calls: tool_calls.clone(),
            tool_calls_count: tool_calls.len(),
            tool_execution_time_ms,
            output_format,
            model_used,
            temperature,
            error: None,
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Create an error response
    pub fn error(
        error: String,
        execution_time_ms: u64,
        model_used: String,
        temperature: f32,
        output_format: String,
    ) -> Self {
        Self {
            content: String::new(),
            success: false,
            execution_time_ms,
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            tools_used: Vec::new(),
            tool_calls: Vec::new(),
            tool_calls_count: 0,
            tool_execution_time_ms: 0,
            output_format,
            model_used,
            temperature,
            error: Some(error),
            metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Check if the response was successful
    pub fn is_success(&self) -> bool {
        self.success
    }

    /// Get the error message if any
    pub fn get_error(&self) -> Option<&String> {
        self.error.as_ref()
    }

    /// Get the output content
    pub fn get_output(&self) -> &str {
        &self.content
    }

    /// Calculate tokens per second
    pub fn tokens_per_second(&self) -> f64 {
        if self.execution_time_ms > 0 {
            (self.total_tokens as f64) / (self.execution_time_ms as f64 / 1000.0)
        } else {
            0.0
        }
    }

    /// Estimate cost based on token usage (placeholder implementation)
    pub fn estimated_cost(&self) -> f64 {
        // This would need to be implemented based on actual pricing
        // For now, return a placeholder calculation
        self.total_tokens as f64 * 0.0001
    }
}