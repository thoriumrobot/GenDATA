// Source-based slice around line 755
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder expireAfterWrite(long,TimeUnit)>

   * @param duration the length of time after an entry is created that it should be automatically
   *     removed
   * @param unit the unit that {@code duration} is expressed in
   * @return this {@code CacheBuilder} instance (for chaining)
   * @throws IllegalArgumentException if {@code duration} is negative
   * @throws IllegalStateException if {@link #expireAfterWrite} was already set
   */
  @SuppressWarnings("GoodTime") // should accept a Duration
  @CanIgnoreReturnValue
  public CacheBuilder<K, V> expireAfterWrite(long duration, TimeUnit unit) {
    checkState(
        expireAfterWriteNanos == UNSET_INT,
        "expireAfterWrite was already set to %s ns",
        expireAfterWriteNanos);
    checkArgument(duration >= 0, "duration cannot be negative: %s %s", duration, unit);
    this.expireAfterWriteNanos = unit.toNanos(duration);
    return this;
  }

  @SuppressWarnings("GoodTime") // nanos internally, should be Duration
