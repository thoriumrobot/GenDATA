// Source-based slice around line 136
// Method: <com.google.common.cache.Cache: void putAll(Map)>


  /**
   * Copies all of the mappings from the specified map to the cache. The effect of this call is
   * equivalent to that of calling {@code put(k, v)} on this map once for each mapping from key
   * {@code k} to value {@code v} in the specified map. The behavior of this operation is undefined
   * if the specified map is modified while the operation is in progress.
   *
   * @since 12.0
   */
  void putAll(Map<? extends K, ? extends V> m);

  /** Discards any cached value for key {@code key}. */
  void invalidate(@CompatibleWith("K") Object key);

  /**
   * Discards any cached values for keys {@code keys}.
   *
   * @since 11.0
   */
  // For discussion of <? extends Object>, see getAllPresent.
