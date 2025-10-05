// Source-based slice around line 125
// Method: <com.google.common.cache.CacheLoader: Map loadAll(Iterable)>

   * @param keys the unique, non-null keys whose values should be loaded
   * @return a map from each key in {@code keys} to the value associated with that key; <b>may not
   *     contain null values</b>
   * @throws Exception if unable to load the result
   * @throws InterruptedException if this method is interrupted. {@code InterruptedException} is
   *     treated like any other {@code Exception} in all respects except that, when it is caught,
   *     the thread's interrupt status is set
   * @since 11.0
   */
  public Map<K, V> loadAll(Iterable<? extends K> keys) throws Exception {
    // This will be caught by getAll(), causing it to fall back to multiple calls to
    // LoadingCache.get
    throw new UnsupportedLoadingOperationException();
  }

  /**
   * Returns a cache loader that uses {@code function} to load keys, without supporting either
   * reloading or bulk loading. This allows creating a cache loader using a lambda expression.
   *
   * <p>The returned object is serializable if {@code function} is serializable.
