// Source-based slice around line 119
// Method: <com.google.common.cache.AbstractCache: void invalidateAll(Iterable)>

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
      invalidate(key);
    }
  }

  @Override
  public void invalidateAll() {
    throw new UnsupportedOperationException();
  }

