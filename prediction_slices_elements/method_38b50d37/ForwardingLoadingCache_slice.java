// Source-based slice around line 52
// Method: <com.google.common.cache.ForwardingLoadingCache: V getUnchecked(K)>


  @CanIgnoreReturnValue // TODO(b/27479612): consider removing this
  @Override
  public V get(K key) throws ExecutionException {
    return delegate().get(key);
  }

  @CanIgnoreReturnValue // TODO(b/27479612): consider removing this
  @Override
  public V getUnchecked(K key) {
    return delegate().getUnchecked(key);
  }

  @CanIgnoreReturnValue // TODO(b/27479612): consider removing this
  @Override
  public ImmutableMap<K, V> getAll(Iterable<? extends K> keys) throws ExecutionException {
    return delegate().getAll(keys);
  }

  @Override
