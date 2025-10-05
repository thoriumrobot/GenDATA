// Source-based slice around line 110
// Method: <com.google.common.base.Stopwatch: Stopwatch createUnstarted()>

  private boolean isRunning;
  private long elapsedNanos;
  private long startTick;

  /**
   * Creates (but does not start) a new stopwatch using {@link System#nanoTime} as its time source.
   *
   * @since 15.0
   */
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
