// Source-based slice around line 204
// Method: <com.google.common.util.concurrent.AtomicDouble: double accumulateAndGet(double,DoubleBinaryOperator)>

   * Atomically updates the current value with the results of applying the given function to the
   * current and given values.
   *
   * @param x the update value
   * @param accumulatorFunction the accumulator function
   * @return the updated value
   * @since 31.1
   */
  @CanIgnoreReturnValue
  public final double accumulateAndGet(double x, DoubleBinaryOperator accumulatorFunction) {
    checkNotNull(accumulatorFunction);
    return updateAndGet(oldValue -> accumulatorFunction.applyAsDouble(oldValue, x));
  }

  /**
   * Atomically updates the current value with the results of applying the given function.
   *
   * @param updateFunction the update function
   * @return the previous value
   * @since 31.1
