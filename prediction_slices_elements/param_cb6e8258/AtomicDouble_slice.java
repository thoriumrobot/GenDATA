// Source-based slice around line 217
// Method: <com.google.common.util.concurrent.AtomicDouble: double getAndUpdate(DoubleUnaryOperator)>


  /**
   * Atomically updates the current value with the results of applying the given function.
   *
   * @param updateFunction the update function
   * @return the previous value
   * @since 31.1
   */
  @CanIgnoreReturnValue
  public final double getAndUpdate(DoubleUnaryOperator updateFunction) {
    while (true) {
      long current = value;
      double currentVal = longBitsToDouble(current);
      double nextVal = updateFunction.applyAsDouble(currentVal);
      long next = doubleToRawLongBits(nextVal);
      if (updater.compareAndSet(this, current, next)) {
        return currentVal;
      }
    }
  }
