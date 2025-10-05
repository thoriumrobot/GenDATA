// Source-based slice around line 46
// Method: <com.google.common.cache.ForwardingLoadingCache: V get(K)>


  /** Constructor for use by subclasses. */
  protected ForwardingLoadingCache() {}

  @Override
  protected abstract LoadingCache<K, V> delegate();

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
