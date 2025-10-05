// Source-based slice around line 1004
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder recordStats()>

   * Enable the accumulation of {@link CacheStats} during the operation of the cache. Without this
   * {@link Cache#stats} will return zero for all statistics. Note that recording stats requires
   * bookkeeping to be performed with each operation, and thus imposes a performance penalty on
   * cache operation.
   *
   * @return this {@code CacheBuilder} instance (for chaining)
   * @since 12.0 (previously, stats collection was automatic)
   */
  @CanIgnoreReturnValue
  public CacheBuilder<K, V> recordStats() {
    statsCounterSupplier = CACHE_STATS_COUNTER;
    return this;
  }

  boolean isRecordingStats() {
    return statsCounterSupplier == CACHE_STATS_COUNTER;
  }

  Supplier<? extends StatsCounter> getStatsCounterSupplier() {
    return statsCounterSupplier;
