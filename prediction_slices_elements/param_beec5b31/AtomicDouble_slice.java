// Source-based slice around line 237
// Method: <com.google.common.util.concurrent.AtomicDouble: double updateAndGet(DoubleUnaryOperator)>


  /**
   * Atomically updates the current value with the results of applying the given function.
   *
   * @param updateFunction the update function
   * @return the updated value
   * @since 31.1
   */
  @CanIgnoreReturnValue
  public final double updateAndGet(DoubleUnaryOperator updateFunction) {
    while (true) {
      long current = value;
      double currentVal = longBitsToDouble(current);
      double nextVal = updateFunction.applyAsDouble(currentVal);
      long next = doubleToRawLongBits(nextVal);
      if (updater.compareAndSet(this, current, next)) {
        return nextVal;
      }
    }
  }
