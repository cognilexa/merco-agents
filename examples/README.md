# Streaming Tool Call Examples

This directory contains comprehensive examples demonstrating the streaming tool call capabilities of the `merco-agents` library.

## 🚀 Quick Start

1. **Set up your environment:**
   ```bash
   # Copy the environment template
   cp examples/basic_streaming_tools/env_template examples/basic_streaming_tools/.env
   
   # Add your API key to the .env file
   echo "OPENROUTER_API_KEY=your_api_key_here" >> examples/basic_streaming_tools/.env
   ```

2. **Run an example:**
   ```bash
   cd examples/basic_streaming_tools
   cargo run
   ```

## 📁 Examples Overview

### 1. Basic Streaming Tools (`basic_streaming_tools/`)
**Purpose:** Simple introduction to streaming tool calls
- ✅ Basic tool call execution
- ✅ Simple streaming handler
- ✅ Multiple tool calls in sequence
- **Best for:** Learning the basics

### 2. Advanced Streaming Tools (`advanced_streaming_tools/`)
**Purpose:** Advanced features with custom handlers
- ✅ Colored output and detailed logging
- ✅ Custom streaming handler with statistics
- ✅ Complex multi-tool scenarios
- ✅ Real-time progress indicators
- **Best for:** Production applications

### 3. Error Handling (`streaming_error_handling/`)
**Purpose:** Testing error resilience and edge cases
- ✅ Error tracking and reporting
- ✅ Edge case handling
- ✅ Ambiguous request handling
- ✅ Performance under stress
- **Best for:** Testing robustness

### 4. Performance Testing (`streaming_performance_test/`)
**Purpose:** Performance monitoring and optimization
- ✅ Execution time tracking
- ✅ Throughput measurement
- ✅ Performance ratings
- ✅ Stress testing
- **Best for:** Performance optimization

## 🔧 Key Features Demonstrated

### Streaming Tool Call Flow
```
1. Text streams → Real-time display
2. Tool call detected → "[🔧 Tool Call (Index N)]"
3. Function info → "Function: name\n   Arguments: {...}"
4. JSON validation → Check if complete
5. If complete → "✓ Arguments complete, executing..."
6. Execute tool → Real-time execution
7. Stream result → "📤 Result: {...}"
8. Continue conversation → LLM gets tool results
```

### Custom Streaming Handlers
```rust
impl StreamingHandler for MyHandler {
    fn handle_chunk(&self, chunk: StreamingChunk) {
        // Handle streaming content
    }
    
    fn handle_tool_calls(&self, tool_calls: Vec<ToolCall>) {
        // Handle tool execution results
    }
    
    fn handle_final(&self, response: StreamingResponse) {
        // Handle completion
    }
    
    fn handle_error(&self, error: String) {
        // Handle errors
    }
}
```

## 🎯 Testing Scenarios

### Basic Tests
- Simple weather queries
- Single tool execution
- Basic error handling

### Advanced Tests
- Multi-city weather queries
- Sequential tool calls
- Complex travel planning
- Real-time progress tracking

### Stress Tests
- Multiple simultaneous tool calls
- Long-running conversations
- Error recovery
- Performance monitoring

## 📊 Performance Metrics

The examples track:
- **Chunk processing time** - How fast content streams
- **Tool execution time** - How fast tools run
- **Throughput** - Chunks and tools per second
- **Success rates** - Error handling effectiveness
- **Memory usage** - Resource efficiency

## 🛠️ Available Tools

The examples use these tools:
- `get_weather(location, unit)` - Weather information
- `web_search(query)` - Web search
- `calculate(expression)` - Mathematical calculations

## 🔍 Debugging Tips

1. **Check API keys** - Ensure your API key is set correctly
2. **Monitor logs** - Watch for tool execution messages
3. **Verify JSON** - Tool arguments must be valid JSON
4. **Check network** - Streaming requires stable connection
5. **Review errors** - Error handlers show detailed information

## 🚨 Common Issues

### Tool Calls Not Executing
- Check if JSON arguments are complete
- Verify tool names match exactly
- Ensure API key has tool access

### Streaming Stops
- Check network connection
- Verify API rate limits
- Review error messages

### Performance Issues
- Monitor chunk processing times
- Check tool execution times
- Review throughput metrics

## 📈 Optimization Tips

1. **Batch tool calls** when possible
2. **Use appropriate timeouts** for tools
3. **Monitor memory usage** during long streams
4. **Implement retry logic** for failed tools
5. **Cache tool results** when appropriate

## 🔗 Related Documentation

- [Streaming API Documentation](../../src/agent/streaming.rs)
- [Agent Execution Documentation](../../src/agent/agent_execution.rs)
- [Tool Call Documentation](../../src/agent/agent.rs)

## 🤝 Contributing

To add new examples:
1. Create a new directory under `examples/`
2. Add `Cargo.toml` with dependencies
3. Implement your example in `src/main.rs`
4. Add an `env_template` file
5. Update this README

## 📝 License

These examples are part of the `merco-agents` project and follow the same license terms.