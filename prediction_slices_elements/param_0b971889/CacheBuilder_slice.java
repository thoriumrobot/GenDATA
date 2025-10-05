// Source-based slice around line 494
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder maximumSize(long)>

   *
   * <p>This feature cannot be used in conjunction with {@link #maximumWeight}.
   *
   * @param maximumSize the maximum size of the cache
   * @return this {@code CacheBuilder} instance (for chaining)
   * @throws IllegalArgumentException if {@code maximumSize} is negative
   * @throws IllegalStateException if a maximum size or weight was already set
   */
  @CanIgnoreReturnValue
  public CacheBuilder<K, V> maximumSize(long maximumSize) {
    checkState(
        this.maximumSize == UNSET_INT, "maximum size was already set to %s", this.maximumSize);
    checkState(
        this.maximumWeight == UNSET_INT,
        "maximum weight was already set to %s",
        this.maximumWeight);
    checkState(this.weigher == null, "maximum size can not be combined with weigher");
    checkArgument(maximumSize >= 0, "maximum size must not be negative");
    this.maximumSize = maximumSize;
    return this;
