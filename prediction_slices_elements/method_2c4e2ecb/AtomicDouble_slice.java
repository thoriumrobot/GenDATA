// Source-based slice around line 175
// Method: <com.google.common.util.concurrent.AtomicDouble: double addAndGet(double)>

  }

  /**
   * Atomically adds the given value to the current value.
   *
   * @param delta the value to add
   * @return the updated value
   */
  @CanIgnoreReturnValue
  public final double addAndGet(double delta) {
    return accumulateAndGet(delta, Double::sum);
  }

  /**
   * Atomically updates the current value with the results of applying the given function to the
   * current and given values.
   *
   * @param x the update value
   * @param accumulatorFunction the accumulator function
   * @return the previous value
