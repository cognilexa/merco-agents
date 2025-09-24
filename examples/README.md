# Merco-Agents Examples

This directory contains comprehensive examples demonstrating the capabilities of the Merco-Agents library.

## 🚀 Quick Start

1. **Set up your environment:**
   ```bash
   # Copy the environment template
   cp basic_agent/env_template .env
   
   # Edit .env and add your API key
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

2. **Run an example:**
   ```bash
   cd basic_agent
   cargo run
   ```

## 📚 Available Examples

### 1. Basic Agent (`basic_agent/`)
**Purpose**: Introduction to basic agent functionality

**Features Demonstrated**:
- Creating a simple agent
- Basic task execution
- String input method (`call_str`)
- Performance metrics
- Agent state management

**Run**: `cd basic_agent && cargo run`

### 2. Output Formats (`output_formats/`)
**Purpose**: Demonstrates configurable output formats

**Features Demonstrated**:
- Text, JSON, Markdown, and HTML output formats
- Agent format configuration
- Task format override
- Format-specific validation
- Format-specific LLM instructions

**Run**: `cd output_formats && cargo run`

### 3. Tool Usage (`tool_usage/`)
**Purpose**: Shows how to integrate custom tools with agents

**Features Demonstrated**:
- Custom tool creation and registration
- Mathematical calculation tools
- Weather information tools
- Time retrieval tools
- Tool parameter validation
- Agent-tool interaction

**Run**: `cd tool_usage && cargo run`

### 4. Multi-Agent (`multi_agent/`)
**Purpose**: Demonstrates multi-agent collaboration

**Features Demonstrated**:
- Specialized agent roles
- Agent collaboration setup
- Sequential task processing
- Parallel task execution
- Cross-agent communication
- Performance tracking per agent

**Run**: `cd multi_agent && cargo run`

## 🔧 Example Structure

Each example follows this structure:
```
example_name/
├── Cargo.toml          # Dependencies
├── src/main.rs         # Example code
├── env_template        # Environment variables template
└── README.md          # Example-specific documentation
```

## 🛠️ Prerequisites

- **Rust**: Latest stable version
- **API Key**: OpenRouter API key for LLM access
- **Dependencies**: All examples use the same core dependencies

## 📋 Common Patterns

### Basic Agent Creation
```rust
let mut agent = Agent::new(
    llm_config,
    "Your agent description".to_string(),
    vec!["Goal 1".to_string(), "Goal 2".to_string()],
    vec![], // tools
);
```

### Task Execution
```rust
let task = Task::new(
    "Your task description".to_string(),
    Some("Expected output description".to_string()),
);

let result = agent.call(task).await?;
```

### Output Format Configuration
```rust
let agent = Agent::new_with_output_format(
    llm_config,
    backstory,
    goals,
    tools,
    OutputFormat::Json, // or Text, Markdown, Html
);
```

### Tool Integration
```rust
let tools = vec![
    Tool {
        function: ToolFunction {
            name: "my_tool".to_string(),
            description: "Tool description".to_string(),
            parameters: tool_parameters,
        },
    },
];

// Register the tool function
merco_llmproxy::register_tool!("my_tool", my_tool_function);
```

## 🎯 Learning Path

1. **Start with Basic Agent** - Understand core concepts
2. **Try Output Formats** - Learn about format configuration
3. **Explore Tool Usage** - Add external capabilities
4. **Master Multi-Agent** - Coordinate multiple agents

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**: Make sure `OPENROUTER_API_KEY` is set in your `.env` file
2. **Compilation Errors**: Ensure you're in the correct example directory
3. **Tool Registration**: Make sure tool functions are registered before use

### Getting Help

- Check the main library documentation
- Review the example source code
- Ensure all dependencies are properly installed

## 🚀 Next Steps

After running the examples:
1. Modify the examples to suit your needs
2. Create your own custom agents
3. Integrate with your existing applications
4. Explore advanced features like memory and state management

Happy coding! 🎉
