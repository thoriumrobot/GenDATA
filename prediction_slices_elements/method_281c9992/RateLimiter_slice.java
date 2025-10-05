// Source-based slice around line 271
// Method: <com.google.common.util.concurrent.RateLimiter: double getRate()>


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
    }
  }

  abstract double doGetRate();

  /**
   * Acquires a single permit from this {@code RateLimiter}, blocking until the request can be
   * granted. Tells the amount of time slept, if any.
