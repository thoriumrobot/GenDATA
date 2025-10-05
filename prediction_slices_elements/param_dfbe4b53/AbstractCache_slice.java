// Source-based slice around line 110
// Method: <com.google.common.cache.AbstractCache: void invalidate(Object)>

  @Override
  public void cleanUp() {}

  @Override
  public long size() {
    throw new UnsupportedOperationException();
  }

  @Override
  public void invalidate(Object key) {
    throw new UnsupportedOperationException();
  }

  /**
   * @since 11.0
   */
  @Override
  // For discussion of <? extends Object>, see getAllPresent.
  public void invalidateAll(Iterable<? extends Object> keys) {
    for (Object key : keys) {
