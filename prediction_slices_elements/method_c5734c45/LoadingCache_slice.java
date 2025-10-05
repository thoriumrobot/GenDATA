// Source-based slice around line 98
// Method: <com.google.common.cache.LoadingCache: V getUnchecked(K)>

   * <p><b>Warning:</b> this method silently converts checked exceptions to unchecked exceptions,
   * and should not be used with cache loaders which throw checked exceptions. In such cases use
   * {@link #get} instead.
   *
   * @throws UncheckedExecutionException if an exception was thrown while loading the value. (As
   *     explained in the last paragraph above, this should be an unchecked exception only.)
   * @throws ExecutionError if an error was thrown while loading the value
   */
  @CanIgnoreReturnValue // TODO(b/27479612): consider removing this?
  V getUnchecked(K key);

  /**
   * Returns a map of the values associated with {@code keys}, creating or retrieving those values
   * if necessary. The returned map contains entries that were already cached, combined with newly
   * loaded entries; it will never contain null keys or values.
   *
   * <p>Caches loaded by a {@link CacheLoader} will issue a single request to {@link
   * CacheLoader#loadAll} for all keys which are not already present in the cache. All entries
   * returned by {@link CacheLoader#loadAll} will be stored in the cache, over-writing any
   * previously cached values. This method will throw an exception if {@link CacheLoader#loadAll}
