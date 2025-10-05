// Source-based slice around line 63
// Method: <com.google.common.cache.ForwardingLoadingCache: V apply(K)>

  }

  @CanIgnoreReturnValue // TODO(b/27479612): consider removing this
  @Override
  public ImmutableMap<K, V> getAll(Iterable<? extends K> keys) throws ExecutionException {
    return delegate().getAll(keys);
  }

  @Override
  public V apply(K key) {
    return delegate().apply(key);
  }

  @Override
  public void refresh(K key) {
    delegate().refresh(key);
  }

  /**
   * A simplified version of {@link ForwardingLoadingCache} where subclasses can pass in an already
