// Source-based slice around line 1009
// Method: <com.google.common.cache.CacheBuilder: boolean isRecordingStats()>

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
  }

  /**
   * Builds a cache, which either returns an already-loaded value for a given key or atomically
   * computes or retrieves it using the supplied {@code CacheLoader}. If another thread is currently
