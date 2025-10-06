use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Current state of an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub status: AgentStatus,
    pub current_task: Option<String>, // Task ID
    pub active_sessions: Vec<String>, // Session IDs
    pub last_activity: DateTime<Utc>,
    pub performance_metrics: PerformanceMetrics,
    pub error_count: u64,
    pub success_count: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AgentStatus {
    Idle,
    Busy,
    Processing,
    Waiting,
    Error,
    Offline,
    Maintenance,
}

/// Context information for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentContext {
    pub session_id: Option<String>,
    pub user_id: Option<String>,
    pub conversation_history: Vec<ConversationEntry>,
    pub shared_memory: HashMap<String, serde_json::Value>,
    pub preferences: AgentPreferences,
    pub environment: EnvironmentContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationEntry {
    pub timestamp: DateTime<Utc>,
    pub role: ConversationRole,
    pub content: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConversationRole {
    User,
    Agent,
    System,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPreferences {
    pub response_style: ResponseStyle,
    pub detail_level: DetailLevel,
    pub language: String,
    pub timezone: String,
    pub notification_preferences: NotificationPreferences,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResponseStyle {
    Concise,
    Detailed,
    Conversational,
    Formal,
    Technical,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DetailLevel {
    Minimal,
    Standard,
    Comprehensive,
    Expert,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationPreferences {
    pub enable_notifications: bool,
    pub notification_types: Vec<NotificationType>,
    pub frequency: NotificationFrequency,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NotificationType {
    TaskCompletion,
    Error,
    StatusChange,
    Collaboration,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NotificationFrequency {
    Immediate,
    Batched,
    Daily,
    Weekly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentContext {
    pub deployment_environment: DeploymentEnvironment,
    pub resource_limits: ResourceLimits,
    pub security_context: SecurityContext,
    pub network_context: NetworkContext,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DeploymentEnvironment {
    Development,
    Staging,
    Production,
    Testing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_memory_mb: u64,
    pub max_cpu_percent: u8,
    pub max_concurrent_requests: usize,
    pub max_response_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContext {
    pub access_level: AccessLevel,
    pub permissions: Vec<Permission>,
    pub encryption_required: bool,
    pub audit_logging: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AccessLevel {
    Public,
    Internal,
    Restricted,
    Confidential,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Permission {
    Read,
    Write,
    Execute,
    Delete,
    Admin,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkContext {
    pub allowed_domains: Vec<String>,
    pub proxy_settings: Option<ProxySettings>,
    pub rate_limits: RateLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxySettings {
    pub host: String,
    pub port: u16,
    pub authentication: Option<ProxyAuth>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyAuth {
    pub username: String,
    pub password: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimits {
    pub requests_per_minute: u32,
    pub requests_per_hour: u32,
    pub requests_per_day: u32,
}

/// Performance metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_tasks: u64,
    pub successful_tasks: u64,
    pub failed_tasks: u64,
    pub average_response_time_ms: f64,
    pub average_tokens_used: f64,
    pub tool_usage_stats: HashMap<String, ToolUsageStats>,
    pub uptime_seconds: u64,
    pub last_reset: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUsageStats {
    pub usage_count: u64,
    pub success_count: u64,
    pub failure_count: u64,
    pub average_execution_time_ms: f64,
    pub last_used: Option<DateTime<Utc>>,
}

impl AgentState {
    pub fn new() -> Self {
        Self {
            status: AgentStatus::Idle,
            current_task: None,
            active_sessions: Vec::new(),
            last_activity: Utc::now(),
            performance_metrics: PerformanceMetrics::new(),
            error_count: 0,
            success_count: 0,
        }
    }

    pub fn update_status(&mut self, status: AgentStatus) {
        self.status = status;
        self.last_activity = Utc::now();
    }

    pub fn start_task(&mut self, task_id: String) {
        self.current_task = Some(task_id);
        self.update_status(AgentStatus::Processing);
    }

    pub fn complete_task(&mut self, success: bool) {
        self.current_task = None;
        self.update_status(AgentStatus::Idle);
        
        if success {
            self.success_count += 1;
        } else {
            self.error_count += 1;
        }
    }

    pub fn add_session(&mut self, session_id: String) {
        if !self.active_sessions.contains(&session_id) {
            self.active_sessions.push(session_id);
        }
    }

    pub fn remove_session(&mut self, session_id: &str) {
        self.active_sessions.retain(|id| id != session_id);
    }
}

impl AgentContext {
    pub fn new() -> Self {
        Self {
            session_id: None,
            user_id: None,
            conversation_history: Vec::new(),
            shared_memory: HashMap::new(),
            preferences: AgentPreferences::default(),
            environment: EnvironmentContext::default(),
        }
    }

    pub fn add_conversation_entry(&mut self, role: ConversationRole, content: String) {
        let entry = ConversationEntry {
            timestamp: Utc::now(),
            role,
            content,
            metadata: HashMap::new(),
        };
        self.conversation_history.push(entry);
    }

    pub fn store_shared_memory(&mut self, key: String, value: serde_json::Value) {
        self.shared_memory.insert(key, value);
    }

    pub fn get_shared_memory(&self, key: &str) -> Option<&serde_json::Value> {
        self.shared_memory.get(key)
    }
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            total_tasks: 0,
            successful_tasks: 0,
            failed_tasks: 0,
            average_response_time_ms: 0.0,
            average_tokens_used: 0.0,
            tool_usage_stats: HashMap::new(),
            uptime_seconds: 0,
            last_reset: Utc::now(),
        }
    }

    pub fn record_task_completion(&mut self, success: bool, response_time_ms: f64, tokens_used: u32) {
        self.total_tasks += 1;
        if success {
            self.successful_tasks += 1;
        } else {
            self.failed_tasks += 1;
        }

        // Update running averages
        self.average_response_time_ms = 
            (self.average_response_time_ms * (self.total_tasks - 1) as f64 + response_time_ms) / self.total_tasks as f64;
        self.average_tokens_used = 
            (self.average_tokens_used * (self.total_tasks - 1) as f64 + tokens_used as f64) / self.total_tasks as f64;
    }

    pub fn record_tool_usage(&mut self, tool_name: String, success: bool, execution_time_ms: f64) {
        let stats = self.tool_usage_stats.entry(tool_name).or_insert(ToolUsageStats {
            usage_count: 0,
            success_count: 0,
            failure_count: 0,
            average_execution_time_ms: 0.0,
            last_used: None,
        });

        stats.usage_count += 1;
        if success {
            stats.success_count += 1;
        } else {
            stats.failure_count += 1;
        }
        stats.average_execution_time_ms = 
            (stats.average_execution_time_ms * (stats.usage_count - 1) as f64 + execution_time_ms) / stats.usage_count as f64;
        stats.last_used = Some(Utc::now());
    }

    pub fn get_success_rate(&self) -> f64 {
        if self.total_tasks == 0 {
            0.0
        } else {
            self.successful_tasks as f64 / self.total_tasks as f64
        }
    }
}

impl Default for AgentPreferences {
    fn default() -> Self {
        Self {
            response_style: ResponseStyle::Conversational,
            detail_level: DetailLevel::Standard,
            language: "en".to_string(),
            timezone: "UTC".to_string(),
            notification_preferences: NotificationPreferences {
                enable_notifications: true,
                notification_types: vec![NotificationType::TaskCompletion, NotificationType::Error],
                frequency: NotificationFrequency::Immediate,
            },
        }
    }
}

impl Default for EnvironmentContext {
    fn default() -> Self {
        Self {
            deployment_environment: DeploymentEnvironment::Development,
            resource_limits: ResourceLimits {
                max_memory_mb: 1024,
                max_cpu_percent: 80,
                max_concurrent_requests: 10,
                max_response_time_ms: 30000,
            },
            security_context: SecurityContext {
                access_level: AccessLevel::Internal,
                permissions: vec![Permission::Read, Permission::Write],
                encryption_required: false,
                audit_logging: true,
            },
            network_context: NetworkContext {
                allowed_domains: vec![],
                proxy_settings: None,
                rate_limits: RateLimits {
                    requests_per_minute: 60,
                    requests_per_hour: 1000,
                    requests_per_day: 10000,
                },
            },
        }
    }
}





