// Source-based slice around line 76
// Method: <com.google.common.cache.ForwardingCache: void put(K,V)>

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

  /**
   * @since 12.0
   */
  @Override
  public void putAll(Map<? extends K, ? extends V> m) {
    delegate().putAll(m);
  }
