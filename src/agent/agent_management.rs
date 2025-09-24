use crate::agent::role::OutputFormat;
use crate::agent::state::{AgentState, AgentStatus};

use crate::agent::agent::Agent;
use merco_llmproxy::Tool;

impl Agent {
    // Getters - consolidated to avoid duplication
    pub fn get_id(&self) -> &str { &self.id }
    pub fn get_name(&self) -> &str { &self.name }
    pub fn get_role(&self) -> &crate::agent::role::AgentRole { &self.role }
    pub fn get_state(&self) -> &AgentState { &self.state }
    pub fn get_capabilities(&self) -> &crate::agent::role::AgentCapabilities { &self.capabilities }
    pub fn get_tools(&self) -> &[Tool] { &self.tools }

    // Legacy getter for backward compatibility
    pub fn get_agent_id(&self) -> &str { self.get_id() }

    // Output handler configuration methods
    pub fn set_output_format(&mut self, format: OutputFormat) {
        self.output_handler.default_format = format;
    }

    pub fn enable_output_validation(&mut self, enabled: bool) {
        self.output_handler.validation_enabled = enabled;
    }

    pub fn set_output_processor(&mut self, processor: fn(&str) -> String) {
        self.output_handler.post_processing = Some(processor);
    }

    // State management methods
    pub fn start_task(&mut self, task_description: String) {
        self.state.start_task(task_description);
    }

    pub fn complete_task(&mut self, success: bool) {
        self.state.complete_task(success);
    }

    pub fn pause_agent(&mut self) {
        self.state.update_status(AgentStatus::Offline);
    }

    pub fn resume_agent(&mut self) {
        self.state.update_status(AgentStatus::Idle);
    }

    pub fn reset_agent(&mut self) {
        self.state = AgentState::new();
        self.context = crate::agent::state::AgentContext::new();
    }

    // Performance metrics
    pub fn get_performance_metrics(&self) -> &crate::agent::state::PerformanceMetrics {
        &self.state.performance_metrics
    }

    pub fn get_success_rate(&self) -> f64 {
        self.state.performance_metrics.get_success_rate()
    }

    pub fn get_average_response_time(&self) -> f64 {
        self.state.performance_metrics.average_response_time_ms
    }

    pub fn get_total_tasks(&self) -> u64 {
        self.state.performance_metrics.total_tasks
    }

    pub fn get_successful_tasks(&self) -> u64 {
        self.state.performance_metrics.successful_tasks
    }

    pub fn get_failed_tasks(&self) -> u64 {
        self.state.performance_metrics.failed_tasks
    }

    // Context management
    pub fn add_context(&mut self, key: String, value: String) {
        self.context.store_shared_memory(key, serde_json::Value::String(value));
    }

    pub fn get_context(&self, key: &str) -> Option<String> {
        self.context.get_shared_memory(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    pub fn clear_context(&mut self) {
        self.context = crate::agent::state::AgentContext::new();
    }

    pub fn get_all_context(&self) -> &std::collections::HashMap<String, serde_json::Value> {
        &self.context.shared_memory
    }

    // Tool management
    pub fn add_tool(&mut self, tool: Tool) {
        if !self.tools.iter().any(|t| t.name == tool.name) {
            self.tools.push(tool);
        }
    }

    pub fn remove_tool(&mut self, tool_name: &str) {
        self.tools.retain(|t| t.name != tool_name);
    }

    pub fn has_tool(&self, tool_name: &str) -> bool {
        self.tools.iter().any(|t| t.name == tool_name)
    }

    // Status checks
    pub fn is_idle(&self) -> bool {
        self.state.status == AgentStatus::Idle
    }

    pub fn is_busy(&self) -> bool {
        self.state.status == AgentStatus::Busy
    }

    pub fn is_paused(&self) -> bool {
        self.state.status == AgentStatus::Offline
    }

    pub fn is_error(&self) -> bool {
        self.state.status == AgentStatus::Error
    }

    // Capability checks
    pub fn can_handle_format(&self, format: &OutputFormat) -> bool {
        self.capabilities.supported_output_formats.contains(format)
    }

    pub fn can_handle_concurrent_tasks(&self, count: usize) -> bool {
        count <= self.capabilities.max_concurrent_tasks
    }

    // Agent information
    pub fn get_agent_info(&self) -> String {
        format!(
            "Agent: {} ({})\nRole: {}\nStatus: {:?}\nTools: {}\nContext entries: {}",
            self.name,
            self.id,
            self.role.name,
            self.state.status,
            self.tools.len(),
            self.context.shared_memory.len()
        )
    }

    // Update methods
    pub fn update_role(&mut self, new_role: crate::agent::role::AgentRole) {
        self.role = new_role;
    }

    pub fn update_capabilities(&mut self, new_capabilities: crate::agent::role::AgentCapabilities) {
        self.capabilities = new_capabilities;
    }

    pub fn update_description(&mut self, new_description: String) {
        self.description = new_description;
    }

    // Utility methods
    pub fn clone_with_new_id(&self, new_id: String) -> Self {
        let mut cloned = self.clone();
        cloned.id = new_id;
        cloned.state = AgentState::new();
        cloned.context = crate::agent::state::AgentContext::new();
        cloned
    }

    pub fn is_healthy(&self) -> bool {
        self.state.status != AgentStatus::Error
    }

    pub fn get_status_summary(&self) -> String {
        format!(
            "{} - Status: {:?}, Tasks: {}/{}, Success Rate: {:.1}%",
            self.name,
            self.state.status,
            self.state.performance_metrics.successful_tasks,
            self.state.performance_metrics.total_tasks,
            self.get_success_rate() * 100.0
        )
    }
}