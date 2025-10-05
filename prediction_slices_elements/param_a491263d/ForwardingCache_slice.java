// Source-based slice around line 68
// Method: <com.google.common.cache.ForwardingCache: ImmutableMap getAllPresent(Iterable)>


  /**
   * @since 11.0
   */
  @Override
  /*
   * <? extends Object> is mostly the same as <?> to plain Java. But to nullness checkers, they
   * differ: <? extends Object> means "non-null types," while <?> means "all types."
   */
  public ImmutableMap<K, V> getAllPresent(Iterable<? extends Object> keys) {
    return delegate().getAllPresent(keys);
  }

  /**
   * @since 11.0
   */
  @Override
  public void put(K key, V value) {
    delegate().put(key, value);
  }
