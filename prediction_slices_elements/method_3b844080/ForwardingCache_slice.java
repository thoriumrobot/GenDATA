// Source-based slice around line 103
// Method: <com.google.common.cache.ForwardingCache: void invalidateAll()>

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
    return delegate().size();
  }

  @Override
  public CacheStats stats() {
