// Source-based slice around line 98
// Method: <com.google.common.cache.ForwardingCache: void invalidateAll(Iterable)>

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
  }

  @Override
  public void invalidateAll() {
    delegate().invalidateAll();
  }

  @Override
  public long size() {
