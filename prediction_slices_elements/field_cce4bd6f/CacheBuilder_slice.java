// Source-based slice around line 301
// Method: com.google.common.cache.CacheBuilder.refreshNanos

  @Nullable Strength valueStrength;

  @SuppressWarnings("GoodTime") // should be a Duration
  long expireAfterWriteNanos = UNSET_INT;

  @SuppressWarnings("GoodTime") // should be a Duration
  long expireAfterAccessNanos = UNSET_INT;

  @SuppressWarnings("GoodTime") // should be a Duration
  long refreshNanos = UNSET_INT;

  @Nullable Equivalence<Object> keyEquivalence;
  @Nullable Equivalence<Object> valueEquivalence;

  @Nullable RemovalListener<? super K, ? super V> removalListener;
  @Nullable Ticker ticker;

  Supplier<? extends StatsCounter> statsCounterSupplier = NULL_STATS_COUNTER;

  private CacheBuilder() {}
