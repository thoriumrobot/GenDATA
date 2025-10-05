// Source-based slice around line 263
// Method: <com.google.common.util.concurrent.RateLimiter: void doSetRate(double,long)>

   * @throws IllegalArgumentException if {@code permitsPerSecond} is negative or zero
   */
  public final void setRate(double permitsPerSecond) {
    checkArgument(permitsPerSecond > 0.0, "rate must be positive");
    synchronized (mutex()) {
      doSetRate(permitsPerSecond, stopwatch.readMicros());
    }
  }

  abstract void doSetRate(double permitsPerSecond, long nowMicros);

  /**
   * Returns the stable rate (as {@code permits per seconds}) with which this {@code RateLimiter} is
   * configured with. The initial value of this is the same as the {@code permitsPerSecond} argument
   * passed in the factory method that produced this {@code RateLimiter}, and it is only updated
   * after invocations to {@linkplain #setRate}.
   */
  public final double getRate() {
    synchronized (mutex()) {
      return doGetRate();
