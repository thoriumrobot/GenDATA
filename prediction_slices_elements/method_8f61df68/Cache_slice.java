// Source-based slice around line 182
// Method: <com.google.common.cache.Cache: void cleanUp()>

   * concurrent use, but if the cache is modified (including by eviction) after the iterator is
   * created, it is undefined which of the changes (if any) will be reflected in that iterator.
   */
  ConcurrentMap<K, V> asMap();

  /**
   * Performs any pending maintenance operations needed by the cache. Exactly which activities are
   * performed -- if any -- is implementation-dependent.
   */
  void cleanUp();
}
