use moka::sync::Cache;
use tracing::debug;

#[derive(Clone)]
pub struct CacheEntry {
    pub response_text: String,
    pub embedding: Vec<f32>,
}

pub struct SemanticCache {
    entries: Cache<String, CacheEntry>,
    similarity_threshold: f32,
}

impl SemanticCache {
    pub fn new(similarity_threshold: f32) -> Self {
        Self {
            entries: Cache::builder().max_capacity(10_000).build(),
            similarity_threshold,
        }
    }

    /// Fast exact-match check (O(1)) without computing embeddings.
    pub fn get_exact(&self, prompt: &str) -> Option<CacheEntry> {
        self.entries.get(prompt)
    }

    /// Semantic search across all cached entries (O(N)).
    pub fn search_semantic(&self, query_embedding: &[f32]) -> Option<CacheEntry> {
        for (key, entry) in self.entries.iter() {
            let sim = cosine_similarity(query_embedding, &entry.embedding);
            if sim >= self.similarity_threshold {
                debug!("Semantic cache hit for '{}' (sim: {:.4})", key, sim);
                return Some(entry.clone());
            }
        }
        None
    }

    /// Insert a new prompt and its result into the cache.
    pub fn insert(&self, prompt: String, embedding: Vec<f32>, response_text: String) {
        self.entries.insert(
            prompt,
            CacheEntry {
                response_text,
                embedding,
            },
        );
    }
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    // Using iterators for safe element access
    for (val_a, val_b) in a.iter().zip(b.iter()) {
        dot_product += val_a * val_b;
        norm_a += val_a * val_a;
        norm_b += val_b * val_b;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a.sqrt() * norm_b.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < f32::EPSILON);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < f32::EPSILON);

        let d = vec![1.0, 1.0, 0.0];
        // norm of d is sqrt(2)
        // dot product a.d is 1.0
        // sim = 1.0 / sqrt(2) = 0.7071
        assert!((cosine_similarity(&a, &d) - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    }
}
