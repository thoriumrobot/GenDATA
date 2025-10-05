// Source-based slice around line 330
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder from(CacheBuilderSpec)>

    return new CacheBuilder<>();
  }

  /**
   * Constructs a new {@code CacheBuilder} instance with the settings specified in {@code spec}.
   *
   * @since 12.0
   */
  @GwtIncompatible // To be supported
  public static CacheBuilder<Object, Object> from(CacheBuilderSpec spec) {
    return spec.toCacheBuilder().lenientParsing();
  }

  /**
   * Constructs a new {@code CacheBuilder} instance with the settings specified in {@code spec}.
   * This is especially useful for command-line configuration of a {@code CacheBuilder}.
   *
   * @param spec a String in the format specified by {@link CacheBuilderSpec}
   * @since 12.0
   */
