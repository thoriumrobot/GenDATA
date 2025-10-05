// Source-based slice around line 151
// Method: <com.google.common.util.concurrent.AtomicDoubleArray: boolean compareAndSet(int,double,double)>

   * Atomically sets the element at position {@code i} to the given updated value if the current
   * value is <a href="#bitEquals">bitwise equal</a> to the expected value.
   *
   * @param i the index
   * @param expect the expected value
   * @param update the new value
   * @return true if successful. False return indicates that the actual value was not equal to the
   *     expected value.
   */
  public final boolean compareAndSet(int i, double expect, double update) {
    return longs.compareAndSet(i, doubleToRawLongBits(expect), doubleToRawLongBits(update));
  }

  /**
   * Atomically sets the element at position {@code i} to the given updated value if the current
   * value is <a href="#bitEquals">bitwise equal</a> to the expected value.
   *
   * <p>May <a
   * href="http://download.oracle.com/javase/7/docs/api/java/util/concurrent/atomic/package-summary.html#Spurious">
   * fail spuriously</a> and does not provide ordering guarantees, so is only rarely an appropriate
