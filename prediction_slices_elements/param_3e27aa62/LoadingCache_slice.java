// Source-based slice around line 136
// Method: <com.google.common.cache.LoadingCache: V apply(K)>

  /**
   * @deprecated Provided to satisfy the {@code Function} interface; use {@link #get} or {@link
   *     #getUnchecked} instead.
   * @throws UncheckedExecutionException if an exception was thrown while loading the value. (As
   *     described in the documentation for {@link #getUnchecked}, {@code LoadingCache} should be
   *     used as a {@code Function} only with cache loaders that throw only unchecked exceptions.)
   */
  @Deprecated
  @Override
  V apply(K key);

  /**
   * Loads a new value for {@code key}, possibly asynchronously. While the new value is loading the
   * previous value (if any) will continue to be returned by {@code get(key)} unless it is evicted.
   * If the new value is loaded successfully it will replace the previous value in the cache; if an
   * exception is thrown while refreshing the previous value will remain, <i>and the exception will
   * be logged (using {@link java.util.logging.Logger}) and swallowed</i>.
   *
   * <p>Caches loaded by a {@link CacheLoader} will call {@link CacheLoader#reload} if the cache
   * currently contains a value for {@code key}, and {@link CacheLoader#load} otherwise. Loading is
