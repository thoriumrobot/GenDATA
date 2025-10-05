// Source-based slice around line 536
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder maximumWeight(long)>

   *
   * @param maximumWeight the maximum total weight of entries the cache may contain
   * @return this {@code CacheBuilder} instance (for chaining)
   * @throws IllegalArgumentException if {@code maximumWeight} is negative
   * @throws IllegalStateException if a maximum weight or size was already set
   * @since 11.0
   */
  @GwtIncompatible // To be supported
  @CanIgnoreReturnValue
  public CacheBuilder<K, V> maximumWeight(long maximumWeight) {
    checkState(
        this.maximumWeight == UNSET_INT,
        "maximum weight was already set to %s",
        this.maximumWeight);
    checkState(
        this.maximumSize == UNSET_INT, "maximum size was already set to %s", this.maximumSize);
    checkArgument(maximumWeight >= 0, "maximum weight must not be negative");
    this.maximumWeight = maximumWeight;
    return this;
  }
