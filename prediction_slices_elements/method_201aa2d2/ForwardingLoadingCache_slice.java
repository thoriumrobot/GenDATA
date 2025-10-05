// Source-based slice around line 58
// Method: <com.google.common.cache.ForwardingLoadingCache: ImmutableMap getAll(Iterable)>


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
  public V apply(K key) {
    return delegate().apply(key);
  }

  @Override
  public void refresh(K key) {
