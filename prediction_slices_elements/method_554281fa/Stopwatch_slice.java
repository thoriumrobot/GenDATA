// Source-based slice around line 235
// Method: <com.google.common.base.Stopwatch: String toString()>

  @J2ktIncompatible
  @GwtIncompatible
  @J2ObjCIncompatible
  public Duration elapsed() {
    return Duration.ofNanos(elapsedNanos());
  }

  /** Returns a string representation of the current elapsed time. */
  @Override
  public String toString() {
    long nanos = elapsedNanos();

    TimeUnit unit = chooseUnit(nanos);
    double value = (double) nanos / NANOSECONDS.convert(1, unit);

    // Too bad this functionality is not exposed as a regular method call
    return Platform.formatCompact4Digits(value) + " " + abbreviate(unit);
  }

  private static TimeUnit chooseUnit(long nanos) {
