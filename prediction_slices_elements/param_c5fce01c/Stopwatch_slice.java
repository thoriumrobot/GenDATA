// Source-based slice around line 119
// Method: <com.google.common.base.Stopwatch: Stopwatch createUnstarted(Ticker)>

  public static Stopwatch createUnstarted() {
    return new Stopwatch();
  }

  /**
   * Creates (but does not start) a new stopwatch, using the specified time source.
   *
   * @since 15.0
   */
  public static Stopwatch createUnstarted(Ticker ticker) {
    return new Stopwatch(ticker);
  }

  /**
   * Creates (and starts) a new stopwatch using {@link System#nanoTime} as its time source.
   *
   * @since 15.0
   */
  public static Stopwatch createStarted() {
    return new Stopwatch().start();
