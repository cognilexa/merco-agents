pub mod agent;
pub mod task;
pub mod crew;

// Re-export main types for easier access
pub use agent::Agent;
pub use agent::AgentModelConfig;
pub use agent::AgentResponse;
pub use agent::TaskResult;
pub use agent::AgentError;
pub use agent::ToolCall;
pub use agent::OutputFormat;
pub use agent::AgentRole;
pub use agent::AgentCapabilities;
pub use agent::ProcessingMode;
pub use agent::Provider;
pub use agent::LlmConfig;
pub use task::task::Task;
