// Source-based slice around line 169
// Method: <com.google.common.util.concurrent.AtomicDoubleArray: boolean weakCompareAndSet(int,double,double)>

   * href="http://download.oracle.com/javase/7/docs/api/java/util/concurrent/atomic/package-summary.html#Spurious">
   * fail spuriously</a> and does not provide ordering guarantees, so is only rarely an appropriate
   * alternative to {@code compareAndSet}.
   *
   * @param i the index
   * @param expect the expected value
   * @param update the new value
   * @return true if successful
   */
  public final boolean weakCompareAndSet(int i, double expect, double update) {
    return longs.weakCompareAndSet(i, doubleToRawLongBits(expect), doubleToRawLongBits(update));
  }

  /**
   * Atomically adds the given value to the element at index {@code i}.
   *
   * @param i the index
   * @param delta the value to add
   * @return the previous value
   */
