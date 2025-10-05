// Source-based slice around line 58
// Method: <com.google.common.graph.MapRetrievalCache: V getIfCached(Object)>

    if (value != null) {
      addToCache((K) key, value);
    }
    return value;
  }

  // Internal methods (package-visible, but treat as only subclass-visible)

  @Override
  @Nullable V getIfCached(@Nullable Object key) {
    V value = super.getIfCached(key);
    if (value != null) {
      return value;
    }

    // Store a local reference to the cache entry. If the backing map is immutable, this,
    // in combination with immutable cache entries, will ensure a thread-safe cache.
    CacheEntry<K, V> entry;

    // Check cache. We use == on purpose because it's cheaper and a cache miss is ok.
