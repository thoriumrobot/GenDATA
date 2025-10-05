// Source-based slice around line 949
// Method: <com.google.common.cache.CacheBuilder: Ticker getTicker(boolean)>

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
  }

  /**
   * Specifies a listener instance that caches should notify each time an entry is removed for any
   * {@linkplain RemovalCause reason}. Each cache created by this builder will invoke this listener
   * as part of the routine maintenance described in the class documentation above.
