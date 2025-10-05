// Source-based slice around line 309
// Method: com.google.common.cache.CacheBuilder.statsCounterSupplier

  @SuppressWarnings("GoodTime") // should be a Duration
  long refreshNanos = UNSET_INT;

  @Nullable Equivalence<Object> keyEquivalence;
  @Nullable Equivalence<Object> valueEquivalence;

  @Nullable RemovalListener<? super K, ? super V> removalListener;
  @Nullable Ticker ticker;

  Supplier<? extends StatsCounter> statsCounterSupplier = NULL_STATS_COUNTER;

  private CacheBuilder() {}

  /**
   * Constructs a new {@code CacheBuilder} instance with default settings, including strong keys,
   * strong values, and no automatic eviction of any kind.
   *
   * <p>Note that while this return type is {@code CacheBuilder<Object, Object>}, type parameters on
   * the {@link #build} methods allow you to create a cache of any key and value type desired.
   */
