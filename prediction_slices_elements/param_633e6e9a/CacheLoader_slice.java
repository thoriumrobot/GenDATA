// Source-based slice around line 97
// Method: <com.google.common.cache.CacheLoader: ListenableFuture reload(K,V)>

   * @return the future new value associated with {@code key}; <b>must not be null, must not return
   *     null</b>
   * @throws Exception if unable to reload the result
   * @throws InterruptedException if this method is interrupted. {@code InterruptedException} is
   *     treated like any other {@code Exception} in all respects except that, when it is caught,
   *     the thread's interrupt status is set
   * @since 11.0
   */
  @GwtIncompatible // Futures
  public ListenableFuture<V> reload(K key, V oldValue) throws Exception {
    checkNotNull(key);
    checkNotNull(oldValue);
    return immediateFuture(load(key));
  }

  /**
   * Computes or retrieves the values corresponding to {@code keys}. This method is called by {@link
   * LoadingCache#getAll}.
   *
   * <p>If the returned map doesn't contain all requested {@code keys} then the entries it does
