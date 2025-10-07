use crate::task::task::Task;
use merco_llmproxy::{
    ChatMessage, CompletionKind, CompletionRequest,
    execute_tool, traits::ChatMessageRole, StreamContentDelta,
};
use futures_util::StreamExt;
use futures::stream::Stream;
use std::pin::Pin;
use async_stream::stream;

use crate::agent::agent::{Agent, AgentResponse};
use crate::agent::streaming::{StreamingChunk, StreamingHandler, DefaultStreamingHandler};
use serde_json;

impl Agent {
    /// Execute a task and return comprehensive response with metrics
    pub async fn call(&mut self, task: Task) -> AgentResponse {
        let start_time = std::time::Instant::now();
        
        match self.process_task_with_metrics(task.clone()).await {
            Ok((content, input_tokens, output_tokens, tools_used, tool_calls)) => {
                let execution_time = start_time.elapsed();
                
                // Determine output format
                let output_format = format!("{:?}", task.output_format);
                
                let response = AgentResponse::success(
                    content,
                    execution_time.as_millis() as u64,
                    input_tokens,
                    output_tokens,
                    self.llm_config.model_name.clone(),
                    self.llm_config.temperature,
                    tools_used,
                    tool_calls,
                    output_format,
                );
                
                // Update agent performance metrics
                self.update_performance_metrics_from_response(&response);
                response
            }
            Err(error) => {
                let execution_time = start_time.elapsed();
                
                // Determine output format for error case
                let output_format = format!("{:?}", task.output_format);
                
                let response = AgentResponse::error(
                    error,
                    execution_time.as_millis() as u64,
                    self.llm_config.model_name.clone(),
                    self.llm_config.temperature,
                    output_format,
                );
                
                // Update agent performance metrics
                self.update_performance_metrics_from_response(&response);
                response
            }
        }
    }

    /// Execute a task with user context
    pub async fn call_with_user(&mut self, task: Task, _user_id: Option<String>) -> AgentResponse {
        // For now, just call the regular call method
        // User context can be added to the task description if needed
        self.call(task).await
    }

    /// Simple string input method - creates a task internally and returns comprehensive response
    pub async fn call_str(&mut self, input: &str) -> AgentResponse {
        // Create a simple task from the string input
        let task = Task::new(input.to_string(), None);
        
        // Use the enhanced call method
        self.call(task).await
    }

    /// Legacy method for backward compatibility - returns just the content
    pub async fn call_legacy(&mut self, task: Task) -> Result<String, String> {
        let response = self.call(task).await;
        if response.success {
            Ok(response.content)
        } else {
            Err(response.error.unwrap_or("Unknown error".to_string()))
        }
    }

    /// Legacy string method for backward compatibility
    pub async fn call_str_legacy(&mut self, input: &str) -> Result<String, String> {
        let response = self.call_str(input).await;
        if response.success {
            Ok(response.content)
        } else {
            Err(response.error.unwrap_or("Unknown error".to_string()))
        }
    }

    /// Core task processing logic with metrics tracking
    async fn process_task_with_metrics(&self, task: Task) -> Result<(String, u32, u32, Vec<String>, Vec<crate::agent::agent::ToolCall>), String> {
        const MAX_RETRIES: usize = 3;
        let mut tools_used = Vec::new();
        let mut all_tool_calls = Vec::new();
        
        for attempt in 1..=MAX_RETRIES {
            let mut messages = self.build_initial_messages(&task);
            
            let (raw_result, input_tokens, output_tokens, tool_calls) = match self.execute_with_llm_with_metrics(&mut messages).await {
                Ok((result, input_toks, output_toks, used_tools, tool_calls)) => {
                    tools_used.extend(used_tools);
                    all_tool_calls.extend(tool_calls);
                    (result, input_toks, output_toks, all_tool_calls.clone())
                }
                Err(e) => {
                    if attempt == MAX_RETRIES {
                        return Err(format!("LLM execution failed after {} attempts: {}", MAX_RETRIES, e));
                    }
                    continue;
                }
            };

            // Determine which format to use: task format if specified, otherwise agent format
            let task_format = &task.output_format;
            let agent_format = &self.output_handler.default_format;
            
            // Convert task format to role format for comparison
            let task_role_format = self.convert_task_format_to_role_format(task_format);
            let use_format = if &task_role_format != agent_format {
                // Task has different format than agent - use task format
                &task_role_format
            } else {
                // Use agent's default format
                agent_format
            };

            // Use the appropriate format for validation
            match self.output_handler.process_output(&raw_result, Some(use_format)) {
                Ok(processed_result) => return Ok((processed_result, input_tokens, output_tokens, tools_used, tool_calls)),
                Err(validation_error) => {
                    if attempt == MAX_RETRIES {
                        return Err(format!("Output validation failed after {} attempts: {}", MAX_RETRIES, validation_error));
                    }
                    
                    messages.push(ChatMessage::new(
                        ChatMessageRole::User,
                        Some(format!("Your previous response was invalid: {}. Please provide a corrected response in the required format.", validation_error)),
                        None,
                        None,
                    ));
                }
            }
        }
        
        Err("Maximum retry attempts exceeded".to_string())
    }

    /// Core LLM execution logic with metrics tracking
    async fn execute_with_llm_with_metrics(&self, messages: &mut Vec<ChatMessage>) -> Result<(String, u32, u32, Vec<String>, Vec<crate::agent::agent::ToolCall>), String> {
        let mut tools_used = Vec::new();
        let mut tool_calls = Vec::new();
        let mut total_input_tokens = 0;
        let mut total_output_tokens = 0;
        
        loop {
            let request = CompletionRequest::new(
                messages.clone(),
                self.llm_config.model_name.clone(),
                Some(self.llm_config.temperature),
                Some(self.llm_config.max_tokens),
                Some(self.tools.clone()),
            );

            match self.provider.completion(request).await {
                Ok(response) => {
                    // Count tokens from messages and response
                    let input_tokens = self.count_input_tokens(messages);
                    total_input_tokens += input_tokens;
                    
                    match response.kind {
                        CompletionKind::Message { content } => {
                            let output_tokens = self.count_output_tokens(&content);
                            total_output_tokens += output_tokens;
                            return Ok((content, total_input_tokens, total_output_tokens, tools_used, tool_calls));
                        }
                        CompletionKind::ToolCall { tool_calls: llm_tool_calls } => {
                            messages.push(ChatMessage::new(
                                ChatMessageRole::Assistant,
                                None,
                                Some(llm_tool_calls.clone()),
                                None,
                            ));
                            
                            for call in llm_tool_calls {
                                let tool_name = call.function.name.clone();
                                let tool_args = call.function.arguments.clone();
                                tools_used.push(tool_name.clone());
                                
                                // Track tool execution time
                                let tool_start = std::time::Instant::now();
                                let (tool_result_content, tool_error) = match execute_tool(&tool_name, &tool_args) {
                                    Ok(result) => (result, None),
                                    Err(e) => {
                                        eprintln!("Tool Execution Error: {}", e);
                                        (String::new(), Some(e))
                                    }
                                };
                                let tool_execution_time = tool_start.elapsed().as_millis() as u64;
                                
                                // Create detailed tool call information
                                let tool_call = if let Some(error) = tool_error {
                                    crate::agent::agent::ToolCall::with_error(
                                        tool_name.clone(),
                                        tool_args,
                                        error,
                                        tool_execution_time,
                                        "text".to_string(), // Default format
                                    )
                                } else {
                                    crate::agent::agent::ToolCall::new(
                                        tool_name.clone(),
                                        tool_args,
                                        tool_result_content.clone(),
                                        tool_execution_time,
                                        "text".to_string(), // Default format
                                    )
                                };
                                tool_calls.push(tool_call);
                                
                                messages.push(ChatMessage::new(
                                    ChatMessageRole::Tool,
                                    Some(tool_result_content),
                                    None,
                                    Some(call.id),
                                ));
                            }
                        }
                    }
                },
                Err(e) => return Err(e.to_string()),
            }
        }
    }

    /// Count input tokens from messages
    fn count_input_tokens(&self, messages: &[ChatMessage]) -> u32 {
        let total_chars: usize = messages.iter()
            .map(|msg| {
                let content_len = msg.content.as_ref().unwrap_or(&String::new()).len();
                // Add role and formatting overhead
                content_len + 20
            })
            .sum();
        // More accurate estimation: ~3.5 characters per token for English text
        (total_chars as f64 / 3.5) as u32
    }

    /// Count output tokens from response content
    fn count_output_tokens(&self, content: &str) -> u32 {
        // More accurate estimation for output tokens
        (content.len() as f64 / 3.5) as u32
    }

    /// Update performance metrics from AgentResponse
    fn update_performance_metrics_from_response(&mut self, response: &AgentResponse) {
        self.state.performance_metrics.record_task_completion(
            response.success,
            response.execution_time_ms as f64,
            response.total_tokens,
        );
    }

    // ===== STREAMING METHODS =====

    /// Execute a task with streaming response - returns a stream of chunks
    pub async fn call_stream(&mut self, task: Task) -> Pin<Box<dyn Stream<Item = Result<StreamingChunk, String>> + Send + '_>> {
        let handler = DefaultStreamingHandler;
        self.call_stream_with_handler(task, handler).await
    }

    /// Execute a task with streaming response and custom handler - FULL tool call support
    pub async fn call_stream_with_handler<H: StreamingHandler + Send + Sync + 'static>(
        &mut self, 
        task: Task, 
        handler: H
    ) -> Pin<Box<dyn Stream<Item = Result<StreamingChunk, String>> + Send + 'static>> {
        let messages = self.build_initial_messages(&task);
        let provider = self.provider.clone();
        let llm_config = self.llm_config.clone();
        let tools = self.tools.clone();
        
        Box::pin(stream! {
            let mut current_messages = messages;
            let mut accumulated_content = String::new();
            let mut total_tokens = 0;
            let mut tools_used = Vec::new();
            let mut all_tool_calls = Vec::new();
            
            loop {
                let request = CompletionRequest::new(
                    current_messages.clone(),
                    llm_config.model_name.clone(),
                    Some(llm_config.temperature),
                    Some(llm_config.max_tokens),
                    Some(tools.clone()),
                );

                match provider.completion_stream(request).await {
                    Ok(mut stream) => {
                        let mut has_tool_calls = false;
                        let mut pending_tool_calls = Vec::new();
                        
                        while let Some(chunk_result) = stream.next().await {
                            match chunk_result {
                                Ok(chunk) => {
                                    match chunk.delta {
                                        StreamContentDelta::Text(text) => {
                                            accumulated_content.push_str(&text);
                                            
                                            let streaming_chunk = StreamingChunk::new(
                                                text,
                                                false,
                                                accumulated_content.clone(),
                                            );
                                            
                                            handler.handle_chunk(streaming_chunk.clone());
                                            yield Ok(streaming_chunk);
                                        }
                                        StreamContentDelta::ToolCallDelta(tool_call_deltas) => {
                                            // Handle streaming tool calls - accumulate deltas
                                            has_tool_calls = true;
                                            
                                            for delta in tool_call_deltas {
                                                // Check if we have complete function info
                                                if let Some(func) = &delta.function {
                                                    if let (Some(name), Some(args)) = (&func.name, &func.arguments) {
                                                        // Check if this is the first time we see this tool call
                                                        let is_new_tool_call = !tools_used.contains(name);
                                                        
                                                        if is_new_tool_call {
                                                            // Tool call is starting
                                                            if let Some(call_id) = &delta.id {
                                                                handler.handle_tool_call_start(name.clone(), call_id.clone());
                                                            }
                                                        }
                                                        
                                                        // Always stream the current arguments (even if partial)
                                                        if let Some(call_id) = &delta.id {
                                                            handler.handle_tool_call_streaming(
                                                                name.clone(), 
                                                                call_id.clone(), 
                                                                args.clone()
                                                            );
                                                        }
                                                        
                                                        // Check if JSON is complete before executing
                                                        if args.starts_with('{') && args.ends_with('}') {
                                                            match serde_json::from_str::<serde_json::Value>(args) {
                                                                Ok(_) => {
                                                                    // JSON is valid and complete - ready to execute
                                                                    if let Some(call_id) = &delta.id {
                                                                        handler.handle_tool_call_ready(
                                                                            name.clone(), 
                                                                            call_id.clone(), 
                                                                            args.clone()
                                                                        );
                                                                    }
                                                                    
                                                                    tools_used.push(name.clone());
                                                                    
                                                                    // Execute the tool
                                                                    let tool_start = std::time::Instant::now();
                                                                    let (tool_result_content, tool_error) = match execute_tool(name, args) {
                                                                        Ok(result) => (result, None),
                                                                        Err(e) => {
                                                                            eprintln!("Tool Execution Error: {}", e);
                                                                            (String::new(), Some(e))
                                                                        }
                                                                    };
                                                                    let tool_execution_time = tool_start.elapsed().as_millis() as u64;
                                                                    
                                                                    // Notify that tool execution is complete
                                                                    if let Some(call_id) = &delta.id {
                                                                        handler.handle_tool_call_executed(
                                                                            name.clone(),
                                                                            call_id.clone(),
                                                                            tool_result_content.clone(),
                                                                            tool_execution_time
                                                                        );
                                                                    }
                                                                    
                                                                    // Create detailed tool call information
                                                                    let tool_call = if let Some(error) = tool_error {
                                                                        crate::agent::agent::ToolCall::with_error(
                                                                            name.clone(),
                                                                            args.clone(),
                                                                            error,
                                                                            tool_execution_time,
                                                                            "text".to_string(),
                                                                        )
                                                                    } else {
                                                                        crate::agent::agent::ToolCall::new(
                                                                            name.clone(),
                                                                            args.clone(),
                                                                            tool_result_content.clone(),
                                                                            tool_execution_time,
                                                                            "text".to_string(),
                                                                        )
                                                                    };
                                                                    all_tool_calls.push(tool_call);
                                                                    
                                                                    // Store for adding to conversation after stream completes
                                                                    pending_tool_calls.push((delta.id.clone(), tool_result_content));
                                                                }
                                                                Err(_) => {
                                                                    // JSON not complete yet - continue streaming
                                                                    // No need to log anything, just continue
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    
                                    // Handle usage statistics if available
                                    if let Some(usage) = chunk.usage {
                                        total_tokens = usage.total_tokens;
                                    }
                                    
                                    // Handle finish reason
                                    if let Some(reason) = chunk.finish_reason {
                                        if has_tool_calls && !pending_tool_calls.is_empty() {
                                            // Add tool results to conversation and continue
                                            let tool_calls_to_add = pending_tool_calls.clone();
                                            pending_tool_calls.clear(); // Clear for next iteration
                                            
                                            for (tool_call_id, tool_result) in tool_calls_to_add {
                                                current_messages.push(ChatMessage::new(
                                                    ChatMessageRole::Tool,
                                                    Some(tool_result),
                                                    None,
                                                    tool_call_id,
                                                ));
                                            }
                                            
                                            // Notify handler about all tool calls
                                            handler.handle_tool_calls(all_tool_calls.clone());
                                            
                                            // Reset for next iteration
                                            accumulated_content.clear();
                                            has_tool_calls = false;
                                            all_tool_calls.clear();
                                            
                                            // Continue the conversation with tool results
                                            continue;
                                        } else {
                                            // No tool calls, finish normally
                                            let final_chunk = StreamingChunk::final_chunk(
                                                String::new(),
                                                accumulated_content.clone(),
                                                chunk.usage.map(|u| crate::agent::streaming::StreamingUsage {
                                                    prompt_tokens: u.prompt_tokens,
                                                    completion_tokens: u.completion_tokens,
                                                    total_tokens: u.total_tokens,
                                                }),
                                                Some(reason),
                                            );
                                            
                                            handler.handle_chunk(final_chunk.clone());
                                            yield Ok(final_chunk);
                                            return;
                                        }
                                    }
                                }
                                Err(e) => {
                                    yield Err(format!("Stream error: {}", e));
                                    return;
                                }
                            }
                        }
                        
                        // If we exit the loop without a finish reason, return the accumulated content
                        if !has_tool_calls {
                            let final_chunk = StreamingChunk::final_chunk(
                                String::new(),
                                accumulated_content.clone(),
                                None,
                                None,
                            );
                            handler.handle_chunk(final_chunk.clone());
                            yield Ok(final_chunk);
                            return;
                        }
                    }
                    Err(e) => {
                        yield Err(format!("Failed to start streaming: {}", e));
                        return;
                    }
                }
            }
        })
    }

    /// Simple string input method with streaming - returns a stream of chunks
    pub async fn call_str_stream(&mut self, input: &str) -> Pin<Box<dyn Stream<Item = Result<StreamingChunk, String>> + Send + '_>> {
        let task = Task::new(input.to_string(), None);
        self.call_stream(task).await
    }

    /// Simple string input method with streaming and custom handler - returns a stream of chunks
    pub async fn call_str_stream_with_handler<H: StreamingHandler + Send + Sync + 'static>(
        &mut self, 
        input: &str, 
        handler: H
    ) -> Pin<Box<dyn Stream<Item = Result<StreamingChunk, String>> + Send + 'static>> {
        let task = Task::new(input.to_string(), None);
        self.call_stream_with_handler(task, handler).await
    }

}