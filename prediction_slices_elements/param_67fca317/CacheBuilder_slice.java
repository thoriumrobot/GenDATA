// Source-based slice around line 833
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder expireAfterAccess(long,TimeUnit)>

   * @param duration the length of time after an entry is last accessed that it should be
   *     automatically removed
   * @param unit the unit that {@code duration} is expressed in
   * @return this {@code CacheBuilder} instance (for chaining)
   * @throws IllegalArgumentException if {@code duration} is negative
   * @throws IllegalStateException if {@link #expireAfterAccess} was already set
   */
  @SuppressWarnings("GoodTime") // should accept a Duration
  @CanIgnoreReturnValue
  public CacheBuilder<K, V> expireAfterAccess(long duration, TimeUnit unit) {
    checkState(
        expireAfterAccessNanos == UNSET_INT,
        "expireAfterAccess was already set to %s ns",
        expireAfterAccessNanos);
    checkArgument(duration >= 0, "duration cannot be negative: %s %s", duration, unit);
    this.expireAfterAccessNanos = unit.toNanos(duration);
    return this;
  }

  @SuppressWarnings("GoodTime") // nanos internally, should be Duration
