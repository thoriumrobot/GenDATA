// Source-based slice around line 458
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder concurrencyLevel(int)>

   *
   * <p>Note that future implementations may abandon segment locking in favor of more advanced
   * concurrency controls.
   *
   * @return this {@code CacheBuilder} instance (for chaining)
   * @throws IllegalArgumentException if {@code concurrencyLevel} is nonpositive
   * @throws IllegalStateException if a concurrency level was already set
   */
  @CanIgnoreReturnValue
  public CacheBuilder<K, V> concurrencyLevel(int concurrencyLevel) {
    checkState(
        this.concurrencyLevel == UNSET_INT,
        "concurrency level was already set to %s",
        this.concurrencyLevel);
    checkArgument(concurrencyLevel > 0);
    this.concurrencyLevel = concurrencyLevel;
    return this;
  }

  int getConcurrencyLevel() {
