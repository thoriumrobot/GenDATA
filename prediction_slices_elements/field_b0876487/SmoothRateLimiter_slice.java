// Source-based slice around line 324
// Method: com.google.common.util.concurrent.SmoothRateLimiter.stableIntervalMicros

  double storedPermits;

  /** The maximum number of stored permits. */
  double maxPermits;

  /**
   * The interval between two unit requests, at our stable rate. E.g., a stable rate of 5 permits
   * per second has a stable interval of 200ms.
   */
  double stableIntervalMicros;

  /**
   * The time when the next request (no matter its size) will be granted. After granting a request,
   * this is pushed further in the future. Large requests push this further than small requests.
   */
  private long nextFreeTicketMicros = 0L; // could be either in the past or future

  private SmoothRateLimiter(SleepingStopwatch stopwatch) {
    super(stopwatch);
  }
