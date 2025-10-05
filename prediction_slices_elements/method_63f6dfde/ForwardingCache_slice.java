// Source-based slice around line 108
// Method: <com.google.common.cache.ForwardingCache: long size()>

    delegate().invalidateAll(keys);
  }

  @Override
  public void invalidateAll() {
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
