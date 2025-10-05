// Source-based slice around line 337
// Method: <com.google.common.util.concurrent.SmoothRateLimiter: void doSetRate(double,long)>

   * this is pushed further in the future. Large requests push this further than small requests.
   */
  private long nextFreeTicketMicros = 0L; // could be either in the past or future

  private SmoothRateLimiter(SleepingStopwatch stopwatch) {
    super(stopwatch);
  }

  @Override
  final void doSetRate(double permitsPerSecond, long nowMicros) {
    resync(nowMicros);
    double stableIntervalMicros = SECONDS.toMicros(1L) / permitsPerSecond;
    this.stableIntervalMicros = stableIntervalMicros;
    doSetRate(permitsPerSecond, stableIntervalMicros);
  }

  abstract void doSetRate(double permitsPerSecond, double stableIntervalMicros);

  @Override
  final double doGetRate() {
