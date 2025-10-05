// Source-based slice around line 928
// Method: <com.google.common.cache.CacheBuilder: long getRefreshNanos()>

  public CacheBuilder<K, V> refreshAfterWrite(long duration, TimeUnit unit) {
    checkNotNull(unit);
    checkState(refreshNanos == UNSET_INT, "refresh was already set to %s ns", refreshNanos);
    checkArgument(duration > 0, "duration must be positive: %s %s", duration, unit);
    this.refreshNanos = unit.toNanos(duration);
    return this;
  }

  @SuppressWarnings("GoodTime") // nanos internally, should be Duration
  long getRefreshNanos() {
    return (refreshNanos == UNSET_INT) ? DEFAULT_REFRESH_NANOS : refreshNanos;
  }

  /**
   * Specifies a nanosecond-precision time source for this cache. By default, {@link
   * System#nanoTime} is used.
   *
   * <p>The primary intent of this method is to facilitate testing of caches with a fake or mock
   * time source.
   *
