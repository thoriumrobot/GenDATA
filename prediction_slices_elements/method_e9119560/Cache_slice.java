// Source-based slice around line 147
// Method: <com.google.common.cache.Cache: void invalidateAll(Iterable)>

  /** Discards any cached value for key {@code key}. */
  void invalidate(@CompatibleWith("K") Object key);

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
