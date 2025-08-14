use super::super::types::*;
use chrono::Utc;
use std::collections::HashMap;

mod working_memory_tests {
    use super::*;

    #[test]
    fn test_working_memory_entry() {
        let entry = WorkingMemoryEntry {
            content: "Test content".to_string(),
            timestamp: Utc::now(),
            importance: 0.8,
            metadata: HashMap::new(),
        };

        assert_eq!(entry.content, "Test content");
        assert!(entry.importance > 0.0);
    }

    #[test]
    fn test_working_memory_capacity() {
        let mut memory = WorkingMemory::new(2); // Capacity of 2
        
        memory.add_entry(WorkingMemoryEntry {
            content: "First entry".to_string(),
            timestamp: Utc::now(),
            importance: 0.5,
            metadata: HashMap::new(),
        });

        memory.add_entry(WorkingMemoryEntry {
            content: "Second entry".to_string(),
            timestamp: Utc::now(),
            importance: 0.7,
            metadata: HashMap::new(),
        });

        // Adding third entry should remove least important
        memory.add_entry(WorkingMemoryEntry {
            content: "Third entry".to_string(),
            timestamp: Utc::now(),
            importance: 0.9,
            metadata: HashMap::new(),
        });

        assert_eq!(memory.entries().len(), 2);
        assert!(memory.entries().iter().any(|e| e.content == "Second entry"));
        assert!(memory.entries().iter().any(|e| e.content == "Third entry"));
    }
}

mod semantic_memory_tests {
    use super::*;

    #[test]
    fn test_semantic_memory_entry() {
        let mut metadata = HashMap::new();
        metadata.insert("category".to_string(), "test".to_string());

        let entry = SemanticMemoryEntry {
            content: "Semantic fact".to_string(),
            timestamp: Utc::now(),
            confidence: 0.9,
            metadata,
        };

        assert_eq!(entry.content, "Semantic fact");
        assert_eq!(entry.confidence, 0.9);
        assert_eq!(entry.metadata.get("category").unwrap(), "test");
    }
}

mod episodic_memory_tests {
    use super::*;

    #[test]
    fn test_episodic_memory_entry() {
        let mut metadata = HashMap::new();
        metadata.insert("location".to_string(), "test room".to_string());

        let entry = EpisodicMemoryEntry {
            content: "Event description".to_string(),
            timestamp: Utc::now(),
            importance: 0.8,
            metadata,
        };

        assert_eq!(entry.content, "Event description");
        assert_eq!(entry.importance, 0.8);
        assert_eq!(entry.metadata.get("location").unwrap(), "test room");
    }
}

mod procedural_memory_tests {
    use super::*;

    #[test]
    fn test_procedural_memory_entry() {
        let mut metadata = HashMap::new();
        metadata.insert("task_type".to_string(), "test task".to_string());

        let entry = ProceduralMemoryEntry {
            content: "Task procedure".to_string(),
            timestamp: Utc::now(),
            success_rate: 0.95,
            metadata,
        };

        assert_eq!(entry.content, "Task procedure");
        assert_eq!(entry.success_rate, 0.95);
        assert_eq!(entry.metadata.get("task_type").unwrap(), "test task");
    }
} 