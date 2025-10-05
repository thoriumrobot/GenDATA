// Source-based slice around line 229
// Method: <com.google.common.base.Stopwatch: Duration elapsed()>

  /**
   * Returns the current elapsed time shown on this stopwatch as a {@link Duration}. Unlike {@link
   * #elapsed(TimeUnit)}, this method does not lose any precision due to rounding.
   *
   * @since 22.0 (but only since 33.4.0 in the Android flavor)
   */
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
