// Source-based slice around line 239
// Method: <com.google.common.util.concurrent.AtomicDoubleArray: double getAndUpdate(int,DoubleUnaryOperator)>

   * Atomically updates the element at index {@code i} with the results of applying the given
   * function to the current value.
   *
   * @param i the index to update
   * @param updaterFunction the update function
   * @return the previous value
   * @since 31.1
   */
  @CanIgnoreReturnValue
  public final double getAndUpdate(int i, DoubleUnaryOperator updaterFunction) {
    while (true) {
      long current = longs.get(i);
      double currentVal = longBitsToDouble(current);
      double nextVal = updaterFunction.applyAsDouble(currentVal);
      long next = doubleToRawLongBits(nextVal);
      if (longs.compareAndSet(i, current, next)) {
        return currentVal;
      }
    }
  }
