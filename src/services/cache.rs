use anndists::dist::distances::DistCosine;
use hnsw_rs::hnsw::Hnsw;
use moka::sync::Cache;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info};

#[derive(Clone)]
pub struct CacheEntry {
    pub response_text: String,
    pub embedding: Vec<f32>,
}

/// A wrapper around everything needed to perform HNSW semantic search.
pub struct SemanticCache {
    /// Original Moka cache with TTL and capacity limits.
    entries: Cache<String, CacheEntry>,
    similarity_threshold: f32,

    /// HNSW index state (Wrapped in RwLock to allow replacing the whole index when rebuilding)
    index_state: RwLock<HnswState>,
}

struct HnswState {
    /// The HNSW index for cosine similarity on f32 arrays, using a static lifetime.
    /// We use a self-referential structure (lifetime erasure) to store the data and 
    /// the index together, as `hnsw_rs` requires a stable reference for insertions.
    // Using unsafe transmutation for lifetime erasure to keep it self-contained
    index: Hnsw<'static, f32, DistCosine>,

    /// Stable addresses for embeddings. `Box<[f32]>` guarantees the pointer doesn't move when `data_store` grows.
    data_store: Vec<Box<[f32]>>,

    /// Maps from HNSW's `usize` data ID to the semantic string prompt key
    id_map: HashMap<usize, String>,

    /// Incremental ID generator for insertions into HNSW
    next_id: usize,

    /// Keep track of how many insertions have happened. Triggers a rebuild when hitting a threshold.
    total_insertions: usize,
}

impl Drop for HnswState {
    fn drop(&mut self) {
        // Drop index first before data_store is dropped
        let empty_index = Hnsw::new(1, 1, 1, 1, DistCosine);
        let _ = std::mem::replace(&mut self.index, empty_index);
    }
}

// Safety: Hnsw stores references to `data_store`. Since `data_store` owns the allocations
// and it's boxed, the addresses are stable. Both are dropped together in `HnswState::drop`.
// So it is safe to Send and Sync as long as elements are not mutated while referenced.
unsafe impl Send for HnswState {}
unsafe impl Sync for HnswState {}

impl SemanticCache {
    pub fn new(similarity_threshold: f32) -> Arc<Self> {
        let max_capacity = 10_000;
        let max_elements = max_capacity as usize * 2;

        let index = Hnsw::new(
            16,           // M
            max_elements, // max capacity
            16,           // M0
            40,           // ef_construction
            DistCosine,
        );

        Arc::new(Self {
            entries: Cache::builder().max_capacity(max_capacity).build(),
            similarity_threshold,
            index_state: RwLock::new(HnswState {
                index,
                data_store: Vec::with_capacity(max_elements),
                id_map: HashMap::new(),
                next_id: 1,
                total_insertions: 0,
            }),
        })
    }

    pub fn get_exact(&self, prompt: &str) -> Option<CacheEntry> {
        self.entries.get(prompt)
    }

    pub fn search_semantic(&self, query_embedding: &[f32]) -> Option<CacheEntry> {
        let state = self.index_state.read();

        // HNSW distance calculation: Cosine similarity in hnsw_rs might just be 1 - cosine
        // We'll calculate our similarity from the raw entries instead since distance might be
        // internal. Actually it just returns a distance struct.
        let ef_search = 32;
        let neighbors = state.index.search(query_embedding, 10, ef_search);

        for neighbor in neighbors {
            let sim = 1.0 - neighbor.distance;

            if sim < self.similarity_threshold {
                continue;
            }

            if let Some(prompt_key) = state.id_map.get(&neighbor.d_id) {
                // `moka::sync::Cache` returns an `Arc<String>` when iterating, but `.get()` takes `&Q`
                // where `K: Borrow<Q>`.
                if let Some(entry) = self.entries.get(prompt_key) {
                    debug!(
                        "Semantic cache hit via HNSW for '{}' (sim: {:.4})",
                        prompt_key, sim
                    );
                    return Some(entry);
                } else {
                    debug!(
                        "HNSW Tombstone found and skipped (ID: {}, Key: '{}')",
                        neighbor.d_id, prompt_key
                    );
                }
            }
        }

        None
    }

    pub fn insert(self: &Arc<Self>, prompt: String, embedding: Vec<f32>, response_text: String) {
        self.entries.insert(
            prompt.clone(),
            CacheEntry {
                response_text,
                embedding: embedding.clone(),
            },
        );

        let mut needs_rebuild = false;

        {
            let mut state = self.index_state.write();
            let id = state.next_id;
            state.next_id += 1;
            state.total_insertions += 1;

            state.id_map.insert(id, prompt);

            let boxed_slice = embedding.into_boxed_slice();
            // Erase lifetime. pointer is stable because it's on the heap (Box).
            let slice_ptr = boxed_slice.as_ref() as *const [f32];
            let static_slice: &'static [f32] = unsafe { &*slice_ptr };

            state.data_store.push(boxed_slice);

            state.index.insert((static_slice, id));

            if state.total_insertions >= 20_000 {
                needs_rebuild = true;
                state.total_insertions = 0;
            }
        }

        if needs_rebuild {
            let cache_clone = self.clone();
            tokio::spawn(async move {
                cache_clone.rebuild_index().await;
            });
        }
    }

    async fn rebuild_index(&self) {
        info!("Starting HNSW background rebuild to prune tombstones...");
        let max_elements = 20_000;

        let new_index = Hnsw::new(16, max_elements, 16, 40, DistCosine);
        let mut new_id_map = HashMap::new();
        let mut new_data_store = Vec::with_capacity(max_elements);
        let mut new_next_id = 1;
        let mut items_retained = 0;

        for (key, entry) in self.entries.iter() {
            let id = new_next_id;
            new_next_id += 1;
            items_retained += 1;

            // `key` returned by `moka::sync::Cache` might be Arc<String> if it's the 0.12 API.
            // Oh wait, `moka`'s iter yields `(Arc<K>, V)` or `(K, V)`.
            // Let's just convert it `to_string()` directly or `as_ref().clone()`.
            let key_str = key.to_string();

            new_id_map.insert(id, key_str);

            let boxed_slice = entry.embedding.clone().into_boxed_slice();
            let slice_ptr = boxed_slice.as_ref() as *const [f32];
            let static_slice: &'static [f32] = unsafe { &*slice_ptr };

            new_data_store.push(boxed_slice);
            new_index.insert((static_slice, id));
        }

        let new_state = HnswState {
            index: new_index,
            data_store: new_data_store,
            id_map: new_id_map,
            next_id: new_next_id,
            total_insertions: items_retained,
        };

        {
            let mut state = self.index_state.write();
            *state = new_state;
        }

        info!(
            "HNSW background rebuild complete. Retained {} alive elements.",
            items_retained
        );
    }
}
