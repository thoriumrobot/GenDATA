// Source-based slice around line 189
// Method: <com.google.common.util.concurrent.AtomicDouble: double getAndAccumulate(double,DoubleBinaryOperator)>

   * Atomically updates the current value with the results of applying the given function to the
   * current and given values.
   *
   * @param x the update value
   * @param accumulatorFunction the accumulator function
   * @return the previous value
   * @since 31.1
   */
  @CanIgnoreReturnValue
  public final double getAndAccumulate(double x, DoubleBinaryOperator accumulatorFunction) {
    checkNotNull(accumulatorFunction);
    return getAndUpdate(oldValue -> accumulatorFunction.applyAsDouble(oldValue, x));
  }

  /**
   * Atomically updates the current value with the results of applying the given function to the
   * current and given values.
   *
   * @param x the update value
   * @param accumulatorFunction the accumulator function
