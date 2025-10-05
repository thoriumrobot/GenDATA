// Source-based slice around line 41
// Method: <com.google.common.graph.MapRetrievalCache: V get(Object)>

  private transient volatile @Nullable CacheEntry<K, V> cacheEntry1;
  private transient volatile @Nullable CacheEntry<K, V> cacheEntry2;

  MapRetrievalCache(Map<K, V> backingMap) {
    super(backingMap);
  }

  @SuppressWarnings("unchecked") // Safe because we only cast if key is found in map.
  @Override
  @Nullable V get(Object key) {
    checkNotNull(key);
    V value = getIfCached(key);
    if (value != null) {
      return value;
    }

    value = getWithoutCaching(key);
    if (value != null) {
      addToCache((K) key, value);
    }
