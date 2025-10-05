// Source-based slice around line 320
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder newBuilder()>

  private CacheBuilder() {}

  /**
   * Constructs a new {@code CacheBuilder} instance with default settings, including strong keys,
   * strong values, and no automatic eviction of any kind.
   *
   * <p>Note that while this return type is {@code CacheBuilder<Object, Object>}, type parameters on
   * the {@link #build} methods allow you to create a cache of any key and value type desired.
   */
  public static CacheBuilder<Object, Object> newBuilder() {
    return new CacheBuilder<>();
  }

  /**
   * Constructs a new {@code CacheBuilder} instance with the settings specified in {@code spec}.
   *
   * @since 12.0
   */
  @GwtIncompatible // To be supported
  public static CacheBuilder<Object, Object> from(CacheBuilderSpec spec) {
