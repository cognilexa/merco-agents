use crate::agent::role::OutputFormat;
use serde::{Deserialize, Serialize};

/// Output Handler for configurable output processing and validation
#[derive(Debug, Clone)]
pub struct OutputHandler {
    pub default_format: OutputFormat,
    pub validation_enabled: bool,
    pub post_processing: Option<fn(&str) -> String>,
}

impl OutputHandler {
    /// Create a new output handler with the specified default format
    pub fn new(default_format: OutputFormat) -> Self {
        Self {
            default_format,
            validation_enabled: true,
            post_processing: None,
        }
    }

    /// Configure validation settings
    pub fn with_validation(mut self, enabled: bool) -> Self {
        self.validation_enabled = enabled;
        self
    }

    /// Add post-processing function
    pub fn with_post_processing(mut self, processor: fn(&str) -> String) -> Self {
        self.post_processing = Some(processor);
        self
    }

    /// Process and validate output based on configured format
    pub fn process_output(&self, raw_output: &str, expected_format: Option<&OutputFormat>) -> Result<String, String> {
        let format = expected_format.unwrap_or(&self.default_format);
        
        // Apply post-processing if configured
        let processed_output = if let Some(processor) = self.post_processing {
            processor(raw_output)
        } else {
            raw_output.to_string()
        };

        // Validate based on format if validation is enabled
        if self.validation_enabled {
            self.validate_output(&processed_output, format)?;
        }

        Ok(processed_output)
    }

    /// Validate output based on the specified format
    fn validate_output(&self, output: &str, format: &OutputFormat) -> Result<(), String> {
        match format {
            OutputFormat::Text => {
                // Basic text validation - just check it's not empty
                if output.trim().is_empty() {
                    return Err("Output cannot be empty".to_string());
                }
                Ok(())
            }
            OutputFormat::Json => {
                // Validate JSON format - handle markdown code blocks
                let json_content = if output.trim().starts_with("```json") && output.trim().ends_with("```") {
                    // Extract JSON from markdown code block
                    let lines: Vec<&str> = output.trim().lines().collect();
                    if lines.len() > 2 {
                        lines[1..lines.len()-1].join("\n")
                    } else {
                        output.to_string()
                    }
                } else if output.trim().starts_with("```") && output.trim().ends_with("```") {
                    // Extract content from generic code block
                    let lines: Vec<&str> = output.trim().lines().collect();
                    if lines.len() > 2 {
                        lines[1..lines.len()-1].join("\n")
                    } else {
                        output.to_string()
                    }
                } else {
                    output.to_string()
                };
                
                match serde_json::from_str::<serde_json::Value>(&json_content) {
                    Ok(_) => Ok(()),
                    Err(e) => Err(format!("Invalid JSON format: {}. Content: {}", e, json_content)),
                }
            }
            OutputFormat::Markdown => {
                // Basic markdown validation - check for common markdown patterns
                if output.trim().is_empty() {
                    return Err("Markdown output cannot be empty".to_string());
                }
                Ok(())
            }
            OutputFormat::Html => {
                // Basic HTML validation - check for opening/closing tags
                if output.trim().is_empty() {
                    return Err("HTML output cannot be empty".to_string());
                }
                Ok(())
            }
            OutputFormat::MultiModal => {
                // Multi-modal validation - for now just check not empty
                if output.trim().is_empty() {
                    return Err("Multi-modal output cannot be empty".to_string());
                }
                Ok(())
            }
        }
    }

    /// Get the current default format
    pub fn get_default_format(&self) -> &OutputFormat {
        &self.default_format
    }

    /// Check if validation is enabled
    pub fn is_validation_enabled(&self) -> bool {
        self.validation_enabled
    }

    /// Check if post-processing is configured
    pub fn has_post_processing(&self) -> bool {
        self.post_processing.is_some()
    }
}

impl Default for OutputHandler {
    fn default() -> Self {
        Self::new(OutputFormat::Text)
    }
}

/// Output validation result with detailed information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub error_message: Option<String>,
    pub format_detected: Option<OutputFormat>,
    pub processing_time_ms: u64,
}

impl ValidationResult {
    pub fn success() -> Self {
        Self {
            is_valid: true,
            error_message: None,
            format_detected: None,
            processing_time_ms: 0,
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            is_valid: false,
            error_message: Some(message),
            format_detected: None,
            processing_time_ms: 0,
        }
    }
}

/// Output processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub default_format: OutputFormat,
    pub validation_enabled: bool,
    pub max_output_length: Option<usize>,
    pub trim_whitespace: bool,
    pub normalize_line_endings: bool,
}

impl OutputConfig {
    pub fn new(default_format: OutputFormat) -> Self {
        Self {
            default_format,
            validation_enabled: true,
            max_output_length: None,
            trim_whitespace: true,
            normalize_line_endings: true,
        }
    }

    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_output_length = Some(max_length);
        self
    }

    pub fn with_trimming(mut self, enabled: bool) -> Self {
        self.trim_whitespace = enabled;
        self
    }

    pub fn with_line_normalization(mut self, enabled: bool) -> Self {
        self.normalize_line_endings = enabled;
        self
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self::new(OutputFormat::Text)
    }
}
