// Source-based slice around line 185
// Method: <com.google.common.cache.CacheLoader: CacheLoader asyncReloading(CacheLoader,Executor)>

   * Returns a {@code CacheLoader} which wraps {@code loader}, executing calls to {@link
   * CacheLoader#reload} using {@code executor}.
   *
   * <p>This method is useful only when {@code loader.reload} has a synchronous implementation, such
   * as {@linkplain #reload the default implementation}.
   *
   * @since 17.0
   */
  @GwtIncompatible // Executor + Futures
  public static <K, V> CacheLoader<K, V> asyncReloading(
      CacheLoader<K, V> loader, Executor executor) {
    checkNotNull(loader);
    checkNotNull(executor);
    return new CacheLoader<K, V>() {
      @Override
      public V load(K key) throws Exception {
        return loader.load(key);
      }

      @Override
