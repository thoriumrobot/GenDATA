// Source-based slice around line 90
// Method: <com.google.common.graph.MapRetrievalCache: void addToCache(K,V)>

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
    // Slide new entry into first cache position. Drop previous entry in second cache position.
    cacheEntry2 = cacheEntry1;
    cacheEntry1 = entry;
  }

  private static final class CacheEntry<K, V> {
