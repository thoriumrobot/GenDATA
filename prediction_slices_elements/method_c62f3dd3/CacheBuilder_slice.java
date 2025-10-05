// Source-based slice around line 882
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder refreshAfterWrite(Duration)>

   * @throws IllegalStateException if {@link #refreshAfterWrite} was already set
   * @throws ArithmeticException for durations greater than +/- approximately 292 years
   * @since 25.0 (but only since 33.3.0 in the Android <a
   *     href="https://github.com/google/guava#guava-google-core-libraries-for-java">flavor</a>)
   */
  @J2ObjCIncompatible
  @GwtIncompatible // Duration
  @SuppressWarnings("GoodTime") // Duration decomposition
  @CanIgnoreReturnValue
  public CacheBuilder<K, V> refreshAfterWrite(Duration duration) {
    return refreshAfterWrite(toNanosSaturated(duration), NANOSECONDS);
  }

  /**
   * Specifies that active entries are eligible for automatic refresh once a fixed duration has
   * elapsed after the entry's creation, or the most recent replacement of its value. The semantics
   * of refreshes are specified in {@link LoadingCache#refresh}, and are performed by calling {@link
   * CacheLoader#reload}.
   *
   * <p>As the default implementation of {@link CacheLoader#reload} is synchronous, it is
