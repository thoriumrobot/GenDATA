// Source-based slice around line 156
// Method: <com.google.common.cache.LoadingCache: void refresh(K)>

   * asynchronous only if {@link CacheLoader#reload} was overridden with an asynchronous
   * implementation.
   *
   * <p>Returns without doing anything if another thread is currently loading the value for {@code
   * key}. If the cache loader associated with this cache performs refresh asynchronously then this
   * method may return before refresh completes.
   *
   * @since 11.0
   */
  void refresh(K key);

  /**
   * {@inheritDoc}
   *
   * <p><b>Note that although the view <i>is</i> modifiable, no method on the returned map will ever
   * cause entries to be automatically loaded.</b>
   */
  @Override
  ConcurrentMap<K, V> asMap();
}
