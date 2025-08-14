use super::super::{WorkingMemory, MemoryEntry, MemoryType};
use async_trait::async_trait;
use chrono::Utc;
use serde::{Serialize, Deserialize};
use merco_llmproxy::ChatMessage;
use std::collections::{HashMap, VecDeque};

/// Working memory implementation for conversation context
#[derive(Debug, Clone)]
pub struct ConversationMemory {
    messages: VecDeque<ChatMessage>,
    max_messages: usize,
    max_tokens: usize,
    summarized_context: Option<String>,
}

impl ConversationMemory {
    pub fn new(max_messages: usize, max_tokens: usize) -> Self {
        Self {
            messages: VecDeque::new(),
            max_messages,
            max_tokens,
            summarized_context: None,
        }
    }

    /// Estimate token count (rough approximation: 1 token ≈ 4 characters)
    fn estimate_tokens(text: &str) -> usize {
        text.chars().count() / 4
    }

    /// Get current context size in tokens
    fn get_context_size(&self) -> usize {
        let default_content = String::new();
        let messages_size: usize = self.messages
            .iter()
            .map(|msg| {
                let content = msg.content.as_ref().unwrap_or(&default_content);
                Self::estimate_tokens(content)
            })
            .sum();
        
        let summary_size = self.summarized_context
            .as_ref()
            .map(|s| Self::estimate_tokens(s))
            .unwrap_or(0);
        
        messages_size + summary_size
    }

    /// Truncate messages to fit within token limit
    fn truncate_if_needed(&mut self) {
        while self.get_context_size() > self.max_tokens && !self.messages.is_empty() {
            self.messages.pop_front();
        }
        
        // If still too large, we need to summarize
        if self.get_context_size() > self.max_tokens {
            // This would trigger summarization in a real implementation
            println!("Warning: Context exceeds token limit, consider summarization");
        }
    }

    pub fn add_chat_message(&mut self, message: ChatMessage) {
        self.messages.push_back(message);
        
        // Remove oldest messages if we exceed the limit
        while self.messages.len() > self.max_messages {
            self.messages.pop_front();
        }
        
        self.truncate_if_needed();
    }

    pub fn get_recent_messages(&self, count: usize) -> Vec<ChatMessage> {
        self.messages
            .iter()
            .rev()
            .take(count)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    pub fn get_all_messages(&self) -> Vec<ChatMessage> {
        self.messages.iter().cloned().collect()
    }

    pub fn get_context_with_summary(&self) -> String {
        let mut context = String::new();
        
        if let Some(summary) = &self.summarized_context {
            context.push_str("Previous conversation summary:\n");
            context.push_str(summary);
            context.push_str("\n\nRecent messages:\n");
        }
        
        for message in &self.messages {
            if let Some(content) = &message.content {
                context.push_str(&format!("{:?}: {}\n", message.role, content));
            }
        }
        
        context
    }
}

#[async_trait]
impl WorkingMemory for ConversationMemory {
    async fn add_message(&mut self, role: String, content: String) -> Result<(), String> {
        let chat_role = match role.as_str() {
            "user" => merco_llmproxy::traits::ChatMessageRole::User,
            "assistant" => merco_llmproxy::traits::ChatMessageRole::Assistant,
            "system" => merco_llmproxy::traits::ChatMessageRole::System,
            "tool" => merco_llmproxy::traits::ChatMessageRole::Tool,
            _ => return Err(format!("Invalid role: {}", role)),
        };
        
        let message = ChatMessage::new(chat_role, Some(content), None, None);
        self.add_chat_message(message);
        Ok(())
    }

    async fn get_context(&self, max_tokens: usize) -> Result<String, String> {
        let context = self.get_context_with_summary();
        
        // Truncate context if it exceeds max_tokens
        if Self::estimate_tokens(&context) > max_tokens {
            let chars_to_keep = max_tokens * 4; // Rough approximation
            if context.len() > chars_to_keep {
                let truncated = &context[..chars_to_keep.min(context.len())];
                return Ok(format!("{}...[truncated]", truncated));
            }
        }
        
        Ok(context)
    }

    async fn summarize_old_context(&mut self) -> Result<(), String> {
        if self.messages.len() < 10 {
            return Ok(()); // Not enough to summarize
        }
        
        // Take first half of messages for summarization
        let messages_to_summarize: Vec<_> = self.messages
            .iter()
            .take(self.messages.len() / 2)
            .cloned()
            .collect();
        
        // Create a simple summary (in a real implementation, you'd use an LLM)
        let summary = format!(
            "Previous conversation involved {} messages covering topics mentioned {} times. Last significant exchange was about message handling.",
            messages_to_summarize.len(),
            messages_to_summarize.len() / 3
        );
        
        // Remove summarized messages
        for _ in 0..messages_to_summarize.len() {
            self.messages.pop_front();
        }
        
        self.summarized_context = Some(summary);
        Ok(())
    }

    async fn clear(&mut self) -> Result<(), String> {
        self.messages.clear();
        self.summarized_context = None;
        Ok(())
    }
}

/// Memory-aware message buffer with automatic management
#[derive(Debug)]
pub struct SmartMessageBuffer {
    working_memory: ConversationMemory,
    importance_threshold: f32,
}

impl SmartMessageBuffer {
    pub fn new(max_messages: usize, max_tokens: usize, importance_threshold: f32) -> Self {
        Self {
            working_memory: ConversationMemory::new(max_messages, max_tokens),
            importance_threshold,
        }
    }

    /// Add message with importance scoring
    pub async fn add_important_message(&mut self, role: String, content: String, importance: f32) -> Result<(), String> {
        if importance >= self.importance_threshold {
            self.working_memory.add_message(role, content).await
        } else {
            // Store in working memory but mark for early removal
            self.working_memory.add_message(role, format!("[LOW_PRIORITY] {}", content)).await
        }
    }

    /// Get messages above importance threshold
    pub fn get_important_messages(&self) -> Vec<ChatMessage> {
        self.working_memory
            .get_all_messages()
            .into_iter()
            .filter(|msg| {
                !msg.content.as_ref().unwrap_or(&String::new()).starts_with("[LOW_PRIORITY]")
            })
            .collect()
    }

    pub async fn get_context(&self, max_tokens: usize) -> Result<String, String> {
        self.working_memory.get_context(max_tokens).await
    }

    pub async fn auto_summarize_if_needed(&mut self) -> Result<(), String> {
        if self.working_memory.messages.len() > self.working_memory.max_messages * 3 / 4 {
            self.working_memory.summarize_old_context().await
        } else {
            Ok(())
        }
    }
} 