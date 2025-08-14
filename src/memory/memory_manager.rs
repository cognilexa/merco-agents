use super::{MemoryStorage, MemoryEntry, MemoryType, MemoryQuery, MemoryResult, SemanticMemory, ProceduralMemory, EpisodicMemory, WorkingMemory, MemoryConsolidation};
use super::types::working::{ConversationMemory, SmartMessageBuffer};
use super::types::semantic::{VectorSemanticMemory, GraphSemanticMemory};
use super::types::episodic::TemporalEpisodicMemory;
use async_trait::async_trait;
use std::collections::HashMap;
use chrono::{Utc, Duration};
use uuid::Uuid;

/// Unified memory manager implementing agentic RAG patterns
pub struct AgenticMemoryManager {
    working_memory: SmartMessageBuffer,
    semantic_memory: Box<dyn SemanticMemory + Send + Sync>,
    episodic_memory: TemporalEpisodicMemory,
    procedural_memory: ProcedureStore,
    consolidation_enabled: bool,
    last_consolidation: Option<chrono::DateTime<Utc>>,
}

impl std::fmt::Debug for AgenticMemoryManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgenticMemoryManager")
            .field("working_memory", &self.working_memory)
            .field("semantic_memory", &"<SemanticMemory trait object>")
            .field("episodic_memory", &self.episodic_memory)
            .field("procedural_memory", &self.procedural_memory)
            .field("consolidation_enabled", &self.consolidation_enabled)
            .field("last_consolidation", &self.last_consolidation)
            .finish()
    }
}

impl AgenticMemoryManager {
    pub fn new(max_working_messages: usize, max_tokens: usize, embedding_dim: usize) -> Self {
        Self {
            working_memory: SmartMessageBuffer::new(max_working_messages, max_tokens, 0.3),
            semantic_memory: Box::new(GraphSemanticMemory::new(embedding_dim)),
            episodic_memory: TemporalEpisodicMemory::new(embedding_dim),
            procedural_memory: ProcedureStore::new(),
            consolidation_enabled: true,
            last_consolidation: None,
        }
    }

    /// Agentic RAG: Query multiple memory types intelligently
    pub async fn agentic_retrieve(&self, query: &str, user_id: Option<String>, context: &str) -> Result<MemoryResult, String> {
        let start_time = std::time::Instant::now();
        let mut all_entries = Vec::new();

        // Determine which memory types to query based on context analysis
        let memory_strategies = self.analyze_query_intent(query, context);

        for strategy in memory_strategies {
            match strategy {
                QueryStrategy::SemanticKnowledge { max_results, boost_recent } => {
                    let semantic_results = self.semantic_memory.search_knowledge(query, max_results).await?;
                    all_entries.extend(semantic_results);
                }
                QueryStrategy::EpisodicExperience { user_id, max_results } => {
                    if let Some(uid) = user_id.as_ref().or(user_id.as_ref()) {
                        let episodic_results = self.episodic_memory.search_experiences(query, Some(uid.clone())).await?;
                        all_entries.extend(episodic_results.into_iter().take(max_results));
                    }
                }
                QueryStrategy::ProceduralKnowledge { domain } => {
                    let procedure_results = self.procedural_memory.search_procedures(query).await?;
                    all_entries.extend(procedure_results);
                }
                QueryStrategy::RecentContext { max_tokens } => {
                    // Get recent conversation context
                    if let Ok(context) = self.working_memory.get_context(max_tokens).await {
                        if !context.is_empty() {
                            let entry = MemoryEntry {
                                id: "working_context".to_string(),
                                content: context,
                                metadata: HashMap::new(),
                                timestamp: Utc::now(),
                                memory_type: MemoryType::Working,
                                relevance_score: Some(0.8),
                                embeddings: None,
                            };
                            all_entries.push(entry);
                        }
                    }
                }
            }
        }

        // Rerank and deduplicate results
        let final_entries = self.rerank_and_deduplicate(all_entries, query).await?;
        
        let search_time = start_time.elapsed().as_millis() as u64;
        
        Ok(MemoryResult {
            entries: final_entries.clone(),
            total_found: final_entries.len(),
            search_time_ms: search_time,
        })
    }

    /// Analyze query to determine optimal memory retrieval strategy
    fn analyze_query_intent(&self, query: &str, context: &str) -> Vec<QueryStrategy> {
        let mut strategies = Vec::new();
        let query_lower = query.to_lowercase();
        let context_lower = context.to_lowercase();

        // Check for factual knowledge queries
        if query_lower.contains("what is") || query_lower.contains("explain") || query_lower.contains("define") {
            strategies.push(QueryStrategy::SemanticKnowledge { max_results: 5, boost_recent: false });
        }

        // Check for personal/experiential queries
        if query_lower.contains("remember") || query_lower.contains("last time") || query_lower.contains("before") {
            strategies.push(QueryStrategy::EpisodicExperience { user_id: None, max_results: 3 });
        }

        // Check for procedural queries
        if query_lower.contains("how to") || query_lower.contains("steps") || query_lower.contains("process") {
            strategies.push(QueryStrategy::ProceduralKnowledge { domain: None });
        }

        // Always include some recent context for continuity
        strategies.push(QueryStrategy::RecentContext { max_tokens: 500 });

        // Default: semantic search if no specific intent detected
        if strategies.len() == 1 {
            strategies.insert(0, QueryStrategy::SemanticKnowledge { max_results: 3, boost_recent: true });
        }

        strategies
    }

    /// Advanced reranking using multiple signals
    async fn rerank_and_deduplicate(&self, mut entries: Vec<MemoryEntry>, query: &str) -> Result<Vec<MemoryEntry>, String> {
        // Remove exact duplicates
        entries.dedup_by(|a, b| a.content == b.content);

        // Score entries using multiple factors
        let mut scored_entries: Vec<(MemoryEntry, f32)> = Vec::new();
        
        for entry in entries {
            let mut score = entry.relevance_score.unwrap_or(0.5);
            
            // Recency boost
            let hours_ago = Utc::now().signed_duration_since(entry.timestamp).num_hours() as f32;
            let recency_factor = (-hours_ago / 168.0).exp(); // Weekly decay
            score *= 1.0 + recency_factor * 0.2;
            
            // Memory type weights
            match entry.memory_type {
                MemoryType::Working => score *= 1.2, // Prioritize recent context
                MemoryType::Episodic => score *= 1.1, // Slightly boost personal experiences
                MemoryType::Semantic => score *= 1.0, // Base weight
                MemoryType::Procedural => score *= 1.15, // Boost actionable knowledge
            }
            
            // Content relevance (simple keyword matching)
            let query_words: Vec<&str> = query.split_whitespace().collect();
            let content_lower = entry.content.to_lowercase();
            let matches = query_words.iter().filter(|word| content_lower.contains(&word.to_lowercase())).count();
            let keyword_boost = (matches as f32 / query_words.len() as f32) * 0.3;
            score += keyword_boost;
            
            scored_entries.push((entry, score));
        }

        // Sort by score
        scored_entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top entries with diversity (avoid too many from same type)
        let mut final_entries = Vec::new();
        let mut type_counts: HashMap<MemoryType, usize> = HashMap::new();
        
        for (entry, _score) in scored_entries {
            let type_count = type_counts.get(&entry.memory_type).unwrap_or(&0);
            if *type_count < 3 { // Max 3 entries per type
                *type_counts.entry(entry.memory_type.clone()).or_insert(0) += 1;
                final_entries.push(entry);
                
                if final_entries.len() >= 10 { // Max total entries
                    break;
                }
            }
        }

        Ok(final_entries)
    }

    /// Store new information with automatic memory type selection
    pub async fn intelligent_store(&mut self, content: String, metadata: HashMap<String, String>, user_id: Option<String>) -> Result<Vec<String>, String> {
        let mut stored_ids = Vec::new();

        // Analyze content to determine appropriate memory types
        let storage_strategies = self.analyze_content_for_storage(&content, &metadata);

        for strategy in storage_strategies {
            match strategy {
                StorageStrategy::WorkingMemory { role } => {
                    self.working_memory.add_important_message(role, content.clone(), 0.7).await?;
                }
                StorageStrategy::SemanticKnowledge => {
                    let id = self.semantic_memory.store_knowledge(content.clone(), metadata.clone()).await?;
                    stored_ids.push(id);
                }
                StorageStrategy::EpisodicExperience => {
                    if let Some(uid) = &user_id {
                        let id = self.episodic_memory.store_experience(uid.clone(), content.clone(), metadata.clone()).await?;
                        stored_ids.push(id);
                    }
                }
                StorageStrategy::ProceduralKnowledge { name, steps } => {
                    let id = self.procedural_memory.store_procedure(name, steps, metadata.clone()).await?;
                    stored_ids.push(id);
                }
            }
        }

        // Trigger consolidation if needed
        if self.should_consolidate().await {
            self.consolidate_memories(user_id).await?;
        }

        Ok(stored_ids)
    }

    fn analyze_content_for_storage(&self, content: &str, metadata: &HashMap<String, String>) -> Vec<StorageStrategy> {
        let mut strategies = Vec::new();
        let content_lower = content.to_lowercase();

        // Always store in working memory for conversation flow
        let role = metadata.get("role").unwrap_or(&"user".to_string()).clone();
        strategies.push(StorageStrategy::WorkingMemory { role });

        // Check for factual knowledge
        if content_lower.contains("fact:") || content_lower.contains("definition:") || metadata.contains_key("knowledge_type") {
            strategies.push(StorageStrategy::SemanticKnowledge);
        }

        // Check for procedural content
        if content_lower.contains("step") || content_lower.contains("process") || content_lower.contains("how to") {
            // Extract steps (simplified)
            let steps: Vec<String> = content.lines()
                .filter(|line| line.trim().starts_with(&['1', '2', '3', '4', '5', '6', '7', '8', '9'][..]) || line.contains("step"))
                .map(|s| s.to_string())
                .collect();
            
            if !steps.is_empty() {
                let name = metadata.get("procedure_name").unwrap_or(&"unnamed_procedure".to_string()).clone();
                strategies.push(StorageStrategy::ProceduralKnowledge { name, steps });
            }
        }

        // Default: store as episodic experience
        if strategies.len() == 1 { // Only working memory so far
            strategies.push(StorageStrategy::EpisodicExperience);
        }

        strategies
    }

    async fn should_consolidate(&self) -> bool {
        if !self.consolidation_enabled {
            return false;
        }

        match self.last_consolidation {
            Some(last) => Utc::now().signed_duration_since(last) > Duration::hours(1),
            None => true,
        }
    }

    /// Memory consolidation: Move important working memory to long-term storage
    async fn consolidate_memories(&mut self, user_id: Option<String>) -> Result<(), String> {
        println!("Starting memory consolidation...");

        // Consolidate working memory to episodic if user context available
        if let Some(uid) = user_id {
            if let Ok(context) = self.working_memory.get_context(2000).await {
                if !context.is_empty() {
                    let mut metadata = HashMap::new();
                    metadata.insert("source".to_string(), "working_memory_consolidation".to_string());
                    metadata.insert("session_id".to_string(), Uuid::new_v4().to_string());
                    
                    self.episodic_memory.store_experience(uid, context, metadata).await?;
                }
            }
        }

        // Auto-summarize working memory
        self.working_memory.auto_summarize_if_needed().await?;

        self.last_consolidation = Some(Utc::now());
        println!("Memory consolidation completed");
        Ok(())
    }

    /// Get comprehensive context for agent decision-making
    pub async fn get_agent_context(&self, query: &str, user_id: Option<String>) -> Result<String, String> {
        let memory_result = self.agentic_retrieve(query, user_id.clone(), "").await?;
        
        let mut context = String::new();
        context.push_str("=== AGENT MEMORY CONTEXT ===\n\n");

        // Group by memory type for clarity
        let mut grouped: HashMap<MemoryType, Vec<&MemoryEntry>> = HashMap::new();
        for entry in &memory_result.entries {
            grouped.entry(entry.memory_type.clone()).or_insert_with(Vec::new).push(entry);
        }

        for (memory_type, entries) in grouped {
            context.push_str(&format!("--- {:?} Memory ---\n", memory_type));
            for entry in entries.iter().take(3) { // Limit entries per type
                context.push_str(&format!("• {}\n", entry.content));
            }
            context.push('\n');
        }

        // Add user patterns if available
        if let Some(uid) = user_id {
            let patterns = self.episodic_memory.identify_user_patterns(&uid);
            if !patterns.is_empty() {
                context.push_str("--- User Patterns ---\n");
                for (pattern, value) in patterns.iter().take(3) {
                    context.push_str(&format!("• {}: {:.2}\n", pattern, value));
                }
                context.push('\n');
            }
        }

        context.push_str("=== END MEMORY CONTEXT ===\n");
        Ok(context)
    }
}

#[derive(Debug, Clone)]
enum QueryStrategy {
    SemanticKnowledge { max_results: usize, boost_recent: bool },
    EpisodicExperience { user_id: Option<String>, max_results: usize },
    ProceduralKnowledge { domain: Option<String> },
    RecentContext { max_tokens: usize },
}

#[derive(Debug, Clone)]
enum StorageStrategy {
    WorkingMemory { role: String },
    SemanticKnowledge,
    EpisodicExperience,
    ProceduralKnowledge { name: String, steps: Vec<String> },
}

/// Simple procedural memory implementation
#[derive(Debug)]
pub struct ProcedureStore {
    procedures: HashMap<String, Vec<String>>,
}

impl ProcedureStore {
    pub fn new() -> Self {
        Self {
            procedures: HashMap::new(),
        }
    }
}

#[async_trait]
impl ProceduralMemory for ProcedureStore {
    async fn store_procedure(&mut self, name: String, steps: Vec<String>, _metadata: HashMap<String, String>) -> Result<String, String> {
        let id = Uuid::new_v4().to_string();
        self.procedures.insert(name, steps);
        Ok(id)
    }

    async fn get_procedure(&self, name: &str) -> Result<Option<Vec<String>>, String> {
        Ok(self.procedures.get(name).cloned())
    }

    async fn search_procedures(&self, query: &str) -> Result<Vec<MemoryEntry>, String> {
        let mut results = Vec::new();
        
        for (name, steps) in &self.procedures {
            if name.to_lowercase().contains(&query.to_lowercase()) {
                let content = format!("Procedure: {}\nSteps:\n{}", name, steps.join("\n"));
                let entry = MemoryEntry {
                    id: Uuid::new_v4().to_string(),
                    content,
                    metadata: HashMap::new(),
                    timestamp: Utc::now(),
                    memory_type: MemoryType::Procedural,
                    relevance_score: Some(0.8),
                    embeddings: None,
                };
                results.push(entry);
            }
        }
        
        Ok(results)
    }
} 