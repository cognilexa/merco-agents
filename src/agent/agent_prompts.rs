use crate::agent::role::OutputFormat;
use crate::agent::agent::Agent;

impl Agent {
    /// Build initial messages for the agent
    pub fn build_initial_messages(&self, task: &crate::task::task::Task) -> Vec<merco_llmproxy::ChatMessage> {
        let system_prompt = self.build_system_prompt();
        let task_prompt = self.build_task_prompt(task);
        
        vec![
            merco_llmproxy::ChatMessage::system(system_prompt),
            merco_llmproxy::ChatMessage::user(task_prompt),
        ]
    }

    /// Build system prompt for the agent
    fn build_system_prompt(&self) -> String {
        format!(
            "You are {}, a specialized AI agent.\n\n\
            ROLE AND CAPABILITIES:\n\
            - Role: {}\n\
            - Description: {}\n\
            - Max Concurrent Tasks: {}\n\
            - Supported Output Formats: {:?}\n\n\
            You have access to the following tools: {}\n\n\
            Always follow the output format specified in the task and provide accurate, helpful responses.",
            self.name,
            self.role.get_description(),
            self.description,
            self.capabilities.max_concurrent_tasks,
            self.capabilities.supported_output_formats,
            self.tools.len()
        )
    }

    fn get_output_format_instruction(&self) -> String {
        self.get_format_instruction(&self.output_handler.default_format)
    }

    fn get_format_instruction(&self, format: &OutputFormat) -> String {
        match format {
            OutputFormat::Text => "Provide your response in plain text format. Be clear and concise.".to_string(),
            OutputFormat::Json => "Provide your response in valid JSON format. Structure your response as a JSON object with appropriate keys and values. Do not wrap your response in markdown code blocks - provide raw JSON only.".to_string(),
            OutputFormat::Markdown => "Provide your response in Markdown format. Use appropriate headers, lists, and formatting.".to_string(),
            OutputFormat::Html => "Provide your response in HTML format. Use proper HTML tags and structure.".to_string(),
            OutputFormat::MultiModal => "Provide your response in a multi-modal format that can include text, images, and other media.".to_string(),
        }
    }

    /// Build task-specific prompt
    fn build_task_prompt(&self, task: &crate::task::task::Task) -> String {
        let mut prompt = format!("Task: {}", task.description);
        
        if let Some(expected_output) = &task.expected_output {
            prompt.push_str(&format!("\nExpected Output: {}", expected_output));
        }
        
        // Always add output format instruction for the task
        let task_role_format = self.convert_task_format_to_role_format(&task.output_format);
        prompt.push_str(&format!("\n\nIMPORTANT - Output Format: {}", self.get_format_instruction(&task_role_format)));
        
        prompt
    }

    /// Convert task output format to role output format
    pub fn convert_task_format_to_role_format(&self, task_format: &crate::task::task::OutputFormat) -> OutputFormat {
        match task_format {
            crate::task::task::OutputFormat::Text => OutputFormat::Text,
            crate::task::task::OutputFormat::Json { .. } => OutputFormat::Json,
        }
    }
}