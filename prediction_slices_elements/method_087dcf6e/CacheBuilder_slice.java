// Source-based slice around line 943
// Method: <com.google.common.cache.CacheBuilder: CacheBuilder ticker(Ticker)>

   * System#nanoTime} is used.
   *
   * <p>The primary intent of this method is to facilitate testing of caches with a fake or mock
   * time source.
   *
   * @return this {@code CacheBuilder} instance (for chaining)
   * @throws IllegalStateException if a ticker was already set
   */
  @CanIgnoreReturnValue
  public CacheBuilder<K, V> ticker(Ticker ticker) {
    checkState(this.ticker == null);
    this.ticker = checkNotNull(ticker);
    return this;
  }

  Ticker getTicker(boolean recordsTime) {
    if (ticker != null) {
      return ticker;
    }
    return recordsTime ? Ticker.systemTicker() : NULL_TICKER;
