use super::super::{EpisodicMemory, MemoryEntry, MemoryType, MemoryQuery};
use async_trait::async_trait;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Episode represents a discrete interaction or experience
#[derive(Debug, Clone)]
pub struct Episode {
    pub id: String,
    pub user_id: String,
    pub session_id: String,
    pub content: String,
    pub emotion: Option<String>,
    pub importance: f32,
    pub context: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
    pub related_episodes: Vec<String>,
}

/// Advanced episodic memory with temporal organization
#[derive(Debug)]
pub struct TemporalEpisodicMemory {
    episodes: HashMap<String, Episode>,
    user_timelines: HashMap<String, Vec<String>>, // user_id -> episode_ids sorted by time
    session_groups: HashMap<String, Vec<String>>, // session_id -> episode_ids
    embedding_dimension: usize,
}

impl TemporalEpisodicMemory {
    pub fn new(embedding_dimension: usize) -> Self {
        Self {
            episodes: HashMap::new(),
            user_timelines: HashMap::new(),
            session_groups: HashMap::new(),
            embedding_dimension,
        }
    }

    /// Create embeddings for episodic content (simplified implementation)
    fn generate_embeddings(&self, text: &str) -> Vec<f32> {
        let mut embeddings = vec![0.0; self.embedding_dimension];
        let bytes = text.as_bytes();
        
        for (i, &byte) in bytes.iter().enumerate() {
            let idx = (i + byte as usize) % self.embedding_dimension;
            embeddings[idx] += (byte as f32) / 255.0;
        }
        
        // Normalize
        let norm: f32 = embeddings.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for embedding in &mut embeddings {
                *embedding /= norm;
            }
        }
        
        embeddings
    }

    /// Calculate importance based on content and context
    fn calculate_importance(&self, content: &str, context: &HashMap<String, String>) -> f32 {
        let mut importance: f32 = 0.5; // Base importance
        
        // Boost importance for certain keywords
        let important_keywords = ["error", "success", "problem", "solution", "learn", "remember"];
        for keyword in &important_keywords {
            if content.to_lowercase().contains(keyword) {
                importance += 0.1;
            }
        }
        
        // Boost based on context
        if context.contains_key("user_feedback") {
            importance += 0.2;
        }
        if context.contains_key("task_completion") {
            importance += 0.15;
        }
        
        importance.min(1.0)
    }

    /// Get episodes within a time window
    pub fn get_episodes_in_timeframe(&self, user_id: &str, start: DateTime<Utc>, end: DateTime<Utc>) -> Vec<Episode> {
        if let Some(episode_ids) = self.user_timelines.get(user_id) {
            episode_ids
                .iter()
                .filter_map(|id| self.episodes.get(id))
                .filter(|episode| episode.timestamp >= start && episode.timestamp <= end)
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get recent episodes with decay-based importance
    pub fn get_recent_important_episodes(&self, user_id: &str, max_count: usize) -> Vec<Episode> {
        let now = Utc::now();
        if let Some(episode_ids) = self.user_timelines.get(user_id) {
            let mut weighted_episodes: Vec<(Episode, f32)> = episode_ids
                .iter()
                .filter_map(|id| self.episodes.get(id))
                .map(|episode| {
                    // Calculate time-weighted importance
                    let hours_ago = now.signed_duration_since(episode.timestamp).num_hours() as f32;
                    let time_decay = (-hours_ago / 168.0).exp(); // Weekly decay
                    let weighted_importance = episode.importance * time_decay;
                    (episode.clone(), weighted_importance)
                })
                .collect();

            weighted_episodes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            weighted_episodes
                .into_iter()
                .take(max_count)
                .map(|(episode, _)| episode)
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Find episodes with similar emotional context
    pub fn get_episodes_by_emotion(&self, user_id: &str, emotion: &str, max_count: usize) -> Vec<Episode> {
        if let Some(episode_ids) = self.user_timelines.get(user_id) {
            episode_ids
                .iter()
                .filter_map(|id| self.episodes.get(id))
                .filter(|episode| {
                    episode.emotion.as_ref().map_or(false, |e| e.eq_ignore_ascii_case(emotion))
                })
                .take(max_count)
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Memory consolidation: identify patterns across episodes
    pub fn identify_user_patterns(&self, user_id: &str) -> HashMap<String, f32> {
        let mut patterns = HashMap::new();
        
        if let Some(episode_ids) = self.user_timelines.get(user_id) {
            let episodes: Vec<_> = episode_ids
                .iter()
                .filter_map(|id| self.episodes.get(id))
                .collect();

            // Analyze temporal patterns
            if episodes.len() > 3 {
                let mut time_gaps = Vec::new();
                for window in episodes.windows(2) {
                    let gap = window[1].timestamp.signed_duration_since(window[0].timestamp).num_hours();
                    time_gaps.push(gap);
                }
                let avg_gap = time_gaps.iter().sum::<i64>() as f32 / time_gaps.len() as f32;
                patterns.insert("avg_interaction_gap_hours".to_string(), avg_gap);
            }

            // Analyze content patterns
            let total_episodes = episodes.len() as f32;
            let error_episodes = episodes.iter().filter(|e| e.content.contains("error")).count() as f32;
            patterns.insert("error_rate".to_string(), error_episodes / total_episodes);

            // Analyze emotional patterns
            let emotions: HashMap<String, usize> = episodes
                .iter()
                .filter_map(|e| e.emotion.as_ref())
                .fold(HashMap::new(), |mut acc, emotion| {
                    *acc.entry(emotion.clone()).or_insert(0) += 1;
                    acc
                });

            for (emotion, count) in emotions {
                patterns.insert(format!("emotion_{}", emotion), count as f32 / total_episodes);
            }
        }
        
        patterns
    }

    /// Automatic episode linking based on similarity
    pub async fn link_related_episodes(&mut self, episode_id: &str) -> Result<(), String> {
        let episode = self.episodes.get(episode_id).ok_or("Episode not found")?.clone();
        let user_episodes: Vec<_> = self.user_timelines
            .get(&episode.user_id)
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|id| self.episodes.get(id))
            .filter(|e| e.id != episode_id)
            .collect();

        let episode_embeddings = self.generate_embeddings(&episode.content);
        let mut similarities = Vec::new();

        for other_episode in user_episodes {
            let other_embeddings = self.generate_embeddings(&other_episode.content);
            let similarity = self.cosine_similarity(&episode_embeddings, &other_embeddings);
            
            if similarity > 0.7 { // High similarity threshold
                similarities.push((other_episode.id.clone(), similarity));
            }
        }

        // Update the episode with related episodes
        if let Some(episode) = self.episodes.get_mut(episode_id) {
            episode.related_episodes = similarities.into_iter().map(|(id, _)| id).collect();
        }

        Ok(())
    }

    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

#[async_trait]
impl EpisodicMemory for TemporalEpisodicMemory {
    async fn store_experience(&mut self, user_id: String, interaction: String, metadata: HashMap<String, String>) -> Result<String, String> {
        let id = Uuid::new_v4().to_string();
        let session_id = metadata.get("session_id").unwrap_or(&"default_session".to_string()).clone();
        let emotion = metadata.get("emotion").cloned();
        let importance = self.calculate_importance(&interaction, &metadata);
        
        let episode = Episode {
            id: id.clone(),
            user_id: user_id.clone(),
            session_id: session_id.clone(),
            content: interaction,
            emotion,
            importance,
            context: metadata,
            timestamp: Utc::now(),
            related_episodes: Vec::new(),
        };

        // Store episode
        self.episodes.insert(id.clone(), episode);

        // Update user timeline
        self.user_timelines
            .entry(user_id)
            .or_insert_with(Vec::new)
            .push(id.clone());

        // Update session grouping
        self.session_groups
            .entry(session_id)
            .or_insert_with(Vec::new)
            .push(id.clone());

        // Link related episodes
        self.link_related_episodes(&id).await?;

        Ok(id)
    }

    async fn get_user_history(&self, user_id: &str, max_results: usize) -> Result<Vec<MemoryEntry>, String> {
        let recent_episodes = self.get_recent_important_episodes(user_id, max_results);
        
        let memory_entries: Vec<MemoryEntry> = recent_episodes
            .into_iter()
            .map(|episode| {
                let embeddings = self.generate_embeddings(&episode.content);
                MemoryEntry {
                    id: episode.id,
                    content: episode.content,
                    metadata: episode.context,
                    timestamp: episode.timestamp,
                    memory_type: MemoryType::Episodic,
                    relevance_score: Some(episode.importance),
                    embeddings: Some(embeddings),
                }
            })
            .collect();

        Ok(memory_entries)
    }

    async fn search_experiences(&self, query: &str, user_id: Option<String>) -> Result<Vec<MemoryEntry>, String> {
        let query_embeddings = self.generate_embeddings(query);
        let mut results = Vec::new();

        let episodes_to_search: Vec<_> = if let Some(uid) = user_id {
            if let Some(episode_ids) = self.user_timelines.get(&uid) {
                episode_ids.iter().filter_map(|id| self.episodes.get(id)).collect()
            } else {
                Vec::new()
            }
        } else {
            self.episodes.values().collect()
        };

        for episode in episodes_to_search {
            let episode_embeddings = self.generate_embeddings(&episode.content);
            let similarity = self.cosine_similarity(&query_embeddings, &episode_embeddings);
            
            if similarity > 0.5 { // Similarity threshold
                let memory_entry = MemoryEntry {
                    id: episode.id.clone(),
                    content: episode.content.clone(),
                    metadata: episode.context.clone(),
                    timestamp: episode.timestamp,
                    memory_type: MemoryType::Episodic,
                    relevance_score: Some(similarity * episode.importance),
                    embeddings: Some(episode_embeddings),
                };
                results.push((memory_entry, similarity));
            }
        }

        // Sort by relevance
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(results.into_iter().map(|(entry, _)| entry).collect())
    }
} 