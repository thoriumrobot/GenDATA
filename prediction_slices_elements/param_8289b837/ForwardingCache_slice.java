// Source-based slice around line 48
// Method: <com.google.common.cache.ForwardingCache: V getIfPresent(Object)>

  protected ForwardingCache() {}

  @Override
  protected abstract Cache<K, V> delegate();

  /**
   * @since 11.0
   */
  @Override
  public @Nullable V getIfPresent(Object key) {
    return delegate().getIfPresent(key);
  }

  /**
   * @since 11.0
   */
  @Override
  public V get(K key, Callable<? extends V> valueLoader) throws ExecutionException {
    return delegate().get(key, valueLoader);
  }
