// Source-based slice around line 73
// Method: <com.google.common.cache.CacheLoader: V load(K)>

   * Computes or retrieves the value corresponding to {@code key}.
   *
   * @param key the non-null key whose value should be loaded
   * @return the value associated with {@code key}; <b>must not be null</b>
   * @throws Exception if unable to load the result
   * @throws InterruptedException if this method is interrupted. {@code InterruptedException} is
   *     treated like any other {@code Exception} in all respects except that, when it is caught,
   *     the thread's interrupt status is set
   */
  public abstract V load(K key) throws Exception;

  /**
   * Computes or retrieves a replacement value corresponding to an already-cached {@code key}. This
   * method is called when an existing cache entry is refreshed by {@link
   * CacheBuilder#refreshAfterWrite}, or through a call to {@link LoadingCache#refresh}.
   *
   * <p>This implementation synchronously delegates to {@link #load}. It is recommended that it be
   * overridden with an asynchronous implementation when using {@link
   * CacheBuilder#refreshAfterWrite}.
   *
