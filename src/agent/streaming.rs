use crate::agent::agent::ToolCall;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use chrono;

/// Streaming response chunk containing incremental content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingChunk {
    /// The incremental content from this chunk
    pub content: String,
    /// Whether this is the final chunk
    pub is_final: bool,
    /// Current accumulated content so far
    pub accumulated_content: String,
    /// Tool call information if this chunk contains tool calls
    pub tool_calls: Option<Vec<crate::agent::agent::ToolCall>>,
    /// Whether this chunk contains tool calls
    pub has_tool_calls: bool,
    /// Usage statistics if available
    pub usage: Option<StreamingUsage>,
    /// Finish reason if available
    pub finish_reason: Option<String>,
    /// Timestamp of this chunk
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Usage statistics for streaming responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Streaming response containing the complete final result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingResponse {
    /// The complete final content
    pub content: String,
    /// Whether the streaming was successful
    pub success: bool,
    /// Total execution time in milliseconds
    pub execution_time_ms: u64,
    /// Total tokens used
    pub total_tokens: u32,
    /// Tools that were used during execution
    pub tools_used: Vec<String>,
    /// Detailed tool calls made
    pub tool_calls: Vec<ToolCall>,
    /// Output format
    pub output_format: String,
    /// Model used
    pub model_used: String,
    /// Temperature setting
    pub temperature: f32,
    /// Any error message if failed
    pub error: Option<String>,
    /// Timestamp when streaming completed
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Callback function type for handling streaming chunks
pub type StreamingCallback = Box<dyn Fn(StreamingChunk) + Send + Sync>;

/// Trait for handling streaming responses with full capability
pub trait StreamingHandler {
    /// Handle a streaming chunk
    fn handle_chunk(&self, chunk: StreamingChunk);
    
    /// Handle tool calls specifically (optional - for enhanced streaming)
    fn handle_tool_calls(&self, tool_calls: Vec<crate::agent::agent::ToolCall>) {
        // Default implementation - do nothing
        let _ = tool_calls;
    }
    
    /// Handle when a tool call starts (tool name is known)
    fn handle_tool_call_start(&self, tool_name: String, call_id: String) {
        // Default implementation - do nothing
        let _ = (tool_name, call_id);
    }
    
    /// Handle streaming tool call parameters as they're being filled
    fn handle_tool_call_streaming(&self, tool_name: String, call_id: String, partial_args: String) {
        // Default implementation - do nothing
        let _ = (tool_name, call_id, partial_args);
    }
    
    /// Handle when tool call parameters are complete and ready to execute
    fn handle_tool_call_ready(&self, tool_name: String, call_id: String, complete_args: String) {
        // Default implementation - do nothing
        let _ = (tool_name, call_id, complete_args);
    }
    
    /// Handle when tool call execution is complete
    fn handle_tool_call_executed(&self, tool_name: String, call_id: String, result: String, execution_time_ms: u64) {
        // Default implementation - do nothing
        let _ = (tool_name, call_id, result, execution_time_ms);
    }
    
    /// Handle the final streaming response
    fn handle_final(&self, response: StreamingResponse);
    
    /// Handle streaming errors
    fn handle_error(&self, error: String);
}

/// Default streaming handler that prints to stdout
pub struct DefaultStreamingHandler;

impl StreamingHandler for DefaultStreamingHandler {
    fn handle_chunk(&self, chunk: StreamingChunk) {
        if !chunk.content.is_empty() {
            print!("{}", chunk.content);
            std::io::stdout().flush().unwrap();
        }
        
        if chunk.has_tool_calls {
            println!("\n[TOOL CALLS DETECTED]");
        }
    }
    
    fn handle_tool_calls(&self, tool_calls: Vec<crate::agent::agent::ToolCall>) {
        println!("\n🔧 Tool Calls:");
        for (i, call) in tool_calls.iter().enumerate() {
            println!("  {}. {} - {}", i + 1, call.tool_name, call.parameters);
        }
    }
    
    fn handle_final(&self, response: StreamingResponse) {
        if let Some(usage) = response.metadata.get("usage") {
            println!("\n\n--- Usage Statistics ---");
            if let Some(prompt_tokens) = usage.get("prompt_tokens") {
                println!("Prompt tokens: {}", prompt_tokens);
            }
            if let Some(completion_tokens) = usage.get("completion_tokens") {
                println!("Completion tokens: {}", completion_tokens);
            }
            if let Some(total_tokens) = usage.get("total_tokens") {
                println!("Total tokens: {}", total_tokens);
            }
        }
        
        if let Some(finish_reason) = response.metadata.get("finish_reason") {
            println!("\n--- Finish Reason: {} ---", finish_reason);
        }
        
        if !response.tools_used.is_empty() {
            println!("Tools used: {:?}", response.tools_used);
        }
        
        println!("\n✅ Streaming completed!");
    }
    
    fn handle_error(&self, error: String) {
        eprintln!("❌ Streaming error: {}", error);
    }
}

impl StreamingResponse {
    /// Create a successful streaming response
    pub fn success(
        content: String,
        execution_time_ms: u64,
        total_tokens: u32,
        tools_used: Vec<String>,
        tool_calls: Vec<ToolCall>,
        output_format: String,
        model_used: String,
        temperature: f32,
    ) -> Self {
        Self {
            content,
            success: true,
            execution_time_ms,
            total_tokens,
            tools_used,
            tool_calls,
            output_format,
            model_used,
            temperature,
            error: None,
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }
    
    /// Create an error streaming response
    pub fn error(
        error: String,
        execution_time_ms: u64,
        output_format: String,
        model_used: String,
        temperature: f32,
    ) -> Self {
        Self {
            content: String::new(),
            success: false,
            execution_time_ms,
            total_tokens: 0,
            tools_used: Vec::new(),
            tool_calls: Vec::new(),
            output_format,
            model_used,
            temperature,
            error: Some(error),
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }
}

impl StreamingChunk {
    /// Create a new streaming chunk
    pub fn new(content: String, is_final: bool, accumulated_content: String) -> Self {
        Self {
            content,
            is_final,
            accumulated_content,
            tool_calls: None,
            has_tool_calls: false,
            usage: None,
            finish_reason: None,
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }
    
    /// Create a chunk with tool calls
    pub fn with_tool_calls(
        content: String,
        is_final: bool,
        accumulated_content: String,
        tool_calls: Vec<crate::agent::agent::ToolCall>,
    ) -> Self {
        Self {
            content,
            is_final,
            accumulated_content,
            tool_calls: Some(tool_calls.clone()),
            has_tool_calls: !tool_calls.is_empty(),
            usage: None,
            finish_reason: None,
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }
    
    /// Create a final chunk with usage statistics
    pub fn final_chunk(
        content: String,
        accumulated_content: String,
        usage: Option<StreamingUsage>,
        finish_reason: Option<String>,
    ) -> Self {
        Self {
            content,
            is_final: true,
            accumulated_content,
            tool_calls: None,
            has_tool_calls: false,
            usage,
            finish_reason,
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }
}
