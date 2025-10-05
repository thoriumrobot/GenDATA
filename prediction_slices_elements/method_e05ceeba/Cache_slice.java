// Source-based slice around line 150
// Method: <com.google.common.cache.Cache: void invalidateAll()>

  /**
   * Discards any cached values for keys {@code keys}.
   *
   * @since 11.0
   */
  // For discussion of <? extends Object>, see getAllPresent.
  void invalidateAll(Iterable<? extends Object> keys);

  /** Discards all entries in the cache. */
  void invalidateAll();

  /** Returns the approximate number of entries in this cache. */
  long size();

  /**
   * Returns a current snapshot of this cache's cumulative statistics, or a set of default values if
   * the cache is not recording statistics. All statistics begin at zero and never decrease over the
   * lifetime of the cache.
   *
   * <p><b>Warning:</b> this cache may not be recording statistical data. For example, a cache
