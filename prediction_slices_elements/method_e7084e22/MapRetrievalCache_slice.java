// Source-based slice around line 84
// Method: <com.google.common.graph.MapRetrievalCache: void clearCache()>

      // Promote second cache entry to first so the access pattern
      // [K1, K2, K1, K3, K1, K4...] still hits the cache half the time.
      addToCache(entry);
      return entry.value;
    }
    return null;
  }

  @Override
  void clearCache() {
    super.clearCache();
    cacheEntry1 = null;
    cacheEntry2 = null;
  }

  private void addToCache(K key, V value) {
    addToCache(new CacheEntry<K, V>(key, value));
  }

  private void addToCache(CacheEntry<K, V> entry) {
