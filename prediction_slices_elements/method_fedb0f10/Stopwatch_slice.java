// Source-based slice around line 137
// Method: <com.google.common.base.Stopwatch: Stopwatch createStarted(Ticker)>

  public static Stopwatch createStarted() {
    return new Stopwatch().start();
  }

  /**
   * Creates (and starts) a new stopwatch, using the specified time source.
   *
   * @since 15.0
   */
  public static Stopwatch createStarted(Ticker ticker) {
    return new Stopwatch(ticker).start();
  }

  Stopwatch() {
    this.ticker = Ticker.systemTicker();
  }

  Stopwatch(Ticker ticker) {
    this.ticker = checkNotNull(ticker, "ticker");
  }
