// Source-based slice around line 919
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder refreshAfterWrite(long,TimeUnit)>

   * @param unit the unit that {@code duration} is expressed in
   * @return this {@code CacheBuilder} instance (for chaining)
   * @throws IllegalArgumentException if {@code duration} is negative
   * @throws IllegalStateException if {@link #refreshAfterWrite} was already set
   * @since 11.0
   */
  @GwtIncompatible // To be supported (synchronously).
  @SuppressWarnings("GoodTime") // should accept a Duration
  @CanIgnoreReturnValue
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
