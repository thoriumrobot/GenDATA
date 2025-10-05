// Source-based slice around line 126
// Method: <com.google.common.cache.AbstractCache: void invalidateAll()>

  @Override
  // For discussion of <? extends Object>, see getAllPresent.
  public void invalidateAll(Iterable<? extends Object> keys) {
    for (Object key : keys) {
      invalidate(key);
    }
  }

  @Override
  public void invalidateAll() {
    throw new UnsupportedOperationException();
  }

  @Override
  public CacheStats stats() {
    throw new UnsupportedOperationException();
  }

  @Override
  public ConcurrentMap<K, V> asMap() {
