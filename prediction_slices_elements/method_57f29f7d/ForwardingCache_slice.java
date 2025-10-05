// Source-based slice around line 113
// Method: <com.google.common.cache.ForwardingCache: CacheStats stats()>

    delegate().invalidateAll();
  }

  @Override
  public long size() {
    return delegate().size();
  }

  @Override
  public CacheStats stats() {
    return delegate().stats();
  }

  @Override
  public ConcurrentMap<K, V> asMap() {
    return delegate().asMap();
  }

  @Override
  public void cleanUp() {
