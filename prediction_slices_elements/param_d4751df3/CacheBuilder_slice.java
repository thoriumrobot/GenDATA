// Source-based slice around line 727
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder expireAfterWrite(Duration)>

   * @throws IllegalStateException if {@link #expireAfterWrite} was already set
   * @throws ArithmeticException for durations greater than +/- approximately 292 years
   * @since 25.0 (but only since 33.3.0 in the Android <a
   *     href="https://github.com/google/guava#guava-google-core-libraries-for-java">flavor</a>)
   */
  @J2ObjCIncompatible
  @GwtIncompatible // Duration
  @SuppressWarnings("GoodTime") // Duration decomposition
  @CanIgnoreReturnValue
  public CacheBuilder<K, V> expireAfterWrite(Duration duration) {
    return expireAfterWrite(toNanosSaturated(duration), NANOSECONDS);
  }

  /**
   * Specifies that each entry should be automatically removed from the cache once a fixed duration
   * has elapsed after the entry's creation, or the most recent replacement of its value.
   *
   * <p>When {@code duration} is zero, this method hands off to {@link #maximumSize(long)
   * maximumSize}{@code (0)}, ignoring any otherwise-specified maximum size or weight. This can be
   * useful in testing, or to disable caching temporarily without a code change.
