// Source-based slice around line 800
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder expireAfterAccess(Duration)>

   * @throws IllegalStateException if {@link #expireAfterAccess} was already set
   * @throws ArithmeticException for durations greater than +/- approximately 292 years
   * @since 25.0 (but only since 33.3.0 in the Android <a
   *     href="https://github.com/google/guava#guava-google-core-libraries-for-java">flavor</a>)
   */
  @J2ObjCIncompatible
  @GwtIncompatible // Duration
  @SuppressWarnings("GoodTime") // Duration decomposition
  @CanIgnoreReturnValue
  public CacheBuilder<K, V> expireAfterAccess(Duration duration) {
    return expireAfterAccess(toNanosSaturated(duration), NANOSECONDS);
  }

  /**
   * Specifies that each entry should be automatically removed from the cache once a fixed duration
   * has elapsed after the entry's creation, the most recent replacement of its value, or its last
   * access. Access time is reset by all cache read and write operations (including {@code
   * Cache.asMap().get(Object)} and {@code Cache.asMap().put(K, V)}), but not by {@code
   * containsKey(Object)}, nor by operations on the collection-views of {@link Cache#asMap}. So, for
   * example, iterating through {@code Cache.asMap().entrySet()} does not reset access time for the
