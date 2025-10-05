// Source-based slice around line 224
// Method: <com.google.common.util.concurrent.AtomicDoubleArray: double accumulateAndGet(int,double,DoubleBinaryOperator)>

   * function to the current and given values.
   *
   * @param i the index to update
   * @param x the update value
   * @param accumulatorFunction the accumulator function
   * @return the updated value
   * @since 31.1
   */
  @CanIgnoreReturnValue
  public final double accumulateAndGet(int i, double x, DoubleBinaryOperator accumulatorFunction) {
    checkNotNull(accumulatorFunction);
    return updateAndGet(i, oldValue -> accumulatorFunction.applyAsDouble(oldValue, x));
  }

  /**
   * Atomically updates the element at index {@code i} with the results of applying the given
   * function to the current value.
   *
   * @param i the index to update
   * @param updaterFunction the update function
