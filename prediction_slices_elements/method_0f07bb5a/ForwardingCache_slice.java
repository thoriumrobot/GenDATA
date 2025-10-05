// Source-based slice around line 89
// Method: <com.google.common.cache.ForwardingCache: void invalidate(Object)>

  /**
   * @since 12.0
   */
  @Override
  public void putAll(Map<? extends K, ? extends V> m) {
    delegate().putAll(m);
  }

  @Override
  public void invalidate(Object key) {
    delegate().invalidate(key);
  }

  /**
   * @since 11.0
   */
  @Override
  // For discussion of <? extends Object>, see getAllPresent.
  public void invalidateAll(Iterable<? extends Object> keys) {
    delegate().invalidateAll(keys);
