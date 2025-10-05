// Source-based slice around line 1013
// Method: <com.google.common.cache.CacheBuilder: Supplier getStatsCounterSupplier()>

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
   * loading the value for this key, simply waits for that thread to finish and returns its loaded
   * value. Note that multiple threads can concurrently load values for distinct keys.
   *
   * <p>This method does not alter the state of this {@code CacheBuilder} instance, so it can be
