// Source-based slice around line 208
// Method: <com.google.common.util.concurrent.AtomicDoubleArray: double getAndAccumulate(int,double,DoubleBinaryOperator)>

   * function to the current and given values.
   *
   * @param i the index to update
   * @param x the update value
   * @param accumulatorFunction the accumulator function
   * @return the previous value
   * @since 31.1
   */
  @CanIgnoreReturnValue
  public final double getAndAccumulate(int i, double x, DoubleBinaryOperator accumulatorFunction) {
    checkNotNull(accumulatorFunction);
    return getAndUpdate(i, oldValue -> accumulatorFunction.applyAsDouble(oldValue, x));
  }

  /**
   * Atomically updates the element at index {@code i} with the results of applying the given
   * function to the current and given values.
   *
   * @param i the index to update
   * @param x the update value
