// Source-based slice around line 342
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder from(String)>


  /**
   * Constructs a new {@code CacheBuilder} instance with the settings specified in {@code spec}.
   * This is especially useful for command-line configuration of a {@code CacheBuilder}.
   *
   * @param spec a String in the format specified by {@link CacheBuilderSpec}
   * @since 12.0
   */
  @GwtIncompatible // To be supported
  public static CacheBuilder<Object, Object> from(String spec) {
    return from(CacheBuilderSpec.parse(spec));
  }

  /**
   * Enables lenient parsing. Useful for tests and spec parsing.
   *
   * @return this {@code CacheBuilder} instance (for chaining)
   */
  @GwtIncompatible // To be supported
  @CanIgnoreReturnValue
