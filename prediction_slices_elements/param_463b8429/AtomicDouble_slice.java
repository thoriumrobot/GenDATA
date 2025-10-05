// Source-based slice around line 135
// Method: <com.google.common.util.concurrent.AtomicDouble: boolean compareAndSet(double,double)>

  /**
   * Atomically sets the value to the given updated value if the current value is <a
   * href="#bitEquals">bitwise equal</a> to the expected value.
   *
   * @param expect the expected value
   * @param update the new value
   * @return {@code true} if successful. False return indicates that the actual value was not
   *     bitwise equal to the expected value.
   */
  public final boolean compareAndSet(double expect, double update) {
    return updater.compareAndSet(this, doubleToRawLongBits(expect), doubleToRawLongBits(update));
  }

  /**
   * Atomically sets the value to the given updated value if the current value is <a
   * href="#bitEquals">bitwise equal</a> to the expected value.
   *
   * <p>May <a
   * href="http://download.oracle.com/javase/7/docs/api/java/util/concurrent/atomic/package-summary.html#Spurious">
   * fail spuriously</a> and does not provide ordering guarantees, so is only rarely an appropriate
