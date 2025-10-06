use crate::agent::agent::{Agent, AgentModelConfig};
use crate::agent::role::{AgentRole, AgentCapabilities, OutputFormat};
use crate::agent::state::AgentState;
use crate::agent::state::AgentContext;
use crate::agent::output_handler::OutputHandler;
use merco_llmproxy::Tool;

impl Agent {
    /// Create a new basic Agent
    pub fn new(
        name: String,
        description: String,
        role: AgentRole,
        llm_config: AgentModelConfig,
        tools: Vec<Tool>,
        capabilities: AgentCapabilities,
    ) -> Self {
        let provider = merco_llmproxy::get_provider(llm_config.to_llmproxy_config()).unwrap();
        
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            description,
            role,
            capabilities,
            llm_config,
            tools,
            state: AgentState::new(),
            context: AgentContext::new(),
            output_handler: OutputHandler::new(OutputFormat::Text),
            provider,
        }
    }

    /// Create a new Agent with custom output format
    pub fn new_with_output_format(
        name: String,
        description: String,
        role: AgentRole,
        llm_config: AgentModelConfig,
        tools: Vec<Tool>,
        capabilities: AgentCapabilities,
        output_format: OutputFormat,
    ) -> Self {
        let provider = merco_llmproxy::get_provider(llm_config.to_llmproxy_config()).unwrap();
        
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            description,
            role,
            capabilities,
            llm_config,
            tools,
            state: AgentState::new(),
            context: AgentContext::new(),
            output_handler: OutputHandler::new(output_format),
            provider,
        }
    }
    
    /// Create a new enhanced Agent with full configuration
    pub fn new_enhanced(
        name: String,
        description: String,
        role: AgentRole,
        llm_config: AgentModelConfig,
        tools: Vec<Tool>,
        capabilities: AgentCapabilities,
        output_format: Option<OutputFormat>,
    ) -> Self {
        let provider = merco_llmproxy::get_provider(llm_config.to_llmproxy_config()).unwrap();
        
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            description,
            role,
            capabilities,
            llm_config,
            tools,
            state: AgentState::new(),
            context: AgentContext::new(),
            output_handler: OutputHandler::new(output_format.unwrap_or(OutputFormat::Text)),
            provider,
        }
    }

    /// Create an Agent with a custom role
    pub fn with_custom_role(
        name: String,
        description: String,
        role: AgentRole,
        llm_config: AgentModelConfig,
        tools: Vec<Tool>,
        capabilities: AgentCapabilities,
        output_format: Option<OutputFormat>,
    ) -> Self {
        let provider = merco_llmproxy::get_provider(llm_config.to_llmproxy_config()).unwrap();
        
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            description,
            role,
            capabilities,
            llm_config,
            tools,
            state: AgentState::new(),
            context: AgentContext::new(),
            output_handler: OutputHandler::new(output_format.unwrap_or(OutputFormat::Text)),
            provider,
        }
    }
}

// Helper trait to convert AgentModelConfig to merco_llmproxy LlmConfig
impl From<AgentModelConfig> for merco_llmproxy::LlmConfig {
    fn from(config: AgentModelConfig) -> Self {
        config.to_llmproxy_config()
    }
}