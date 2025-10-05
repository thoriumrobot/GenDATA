// Source-based slice around line 152
// Method: <com.google.common.util.concurrent.AtomicDouble: boolean weakCompareAndSet(double,double)>

   * <p>May <a
   * href="http://download.oracle.com/javase/7/docs/api/java/util/concurrent/atomic/package-summary.html#Spurious">
   * fail spuriously</a> and does not provide ordering guarantees, so is only rarely an appropriate
   * alternative to {@code compareAndSet}.
   *
   * @param expect the expected value
   * @param update the new value
   * @return {@code true} if successful
   */
  public final boolean weakCompareAndSet(double expect, double update) {
    return updater.weakCompareAndSet(
        this, doubleToRawLongBits(expect), doubleToRawLongBits(update));
  }

  /**
   * Atomically adds the given value to the current value.
   *
   * @param delta the value to add
   * @return the previous value
   */
