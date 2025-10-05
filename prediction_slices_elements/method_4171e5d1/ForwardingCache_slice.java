// Source-based slice around line 56
// Method: <com.google.common.cache.ForwardingCache: V get(K,Callable)>

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

  /**
   * @since 11.0
   */
  @Override
  /*
   * <? extends Object> is mostly the same as <?> to plain Java. But to nullness checkers, they
   * differ: <? extends Object> means "non-null types," while <?> means "all types."
