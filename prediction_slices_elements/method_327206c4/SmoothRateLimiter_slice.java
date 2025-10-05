// Source-based slice around line 347
// Method: <com.google.common.util.concurrent.SmoothRateLimiter: double doGetRate()>

    resync(nowMicros);
    double stableIntervalMicros = SECONDS.toMicros(1L) / permitsPerSecond;
    this.stableIntervalMicros = stableIntervalMicros;
    doSetRate(permitsPerSecond, stableIntervalMicros);
  }

  abstract void doSetRate(double permitsPerSecond, double stableIntervalMicros);

  @Override
  final double doGetRate() {
    return SECONDS.toMicros(1L) / stableIntervalMicros;
  }

  @Override
  final long queryEarliestAvailable(long nowMicros) {
    return nextFreeTicketMicros;
  }

  @Override
  final long reserveEarliestAvailable(int requiredPermits, long nowMicros) {
