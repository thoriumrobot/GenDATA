// Source-based slice around line 193
// Method: <com.google.common.util.concurrent.AtomicDoubleArray: double addAndGet(int,double)>


  /**
   * Atomically adds the given value to the element at index {@code i}.
   *
   * @param i the index
   * @param delta the value to add
   * @return the updated value
   */
  @CanIgnoreReturnValue
  public double addAndGet(int i, double delta) {
    return accumulateAndGet(i, delta, Double::sum);
  }

  /**
   * Atomically updates the element at index {@code i} with the results of applying the given
   * function to the current and given values.
   *
   * @param i the index to update
   * @param x the update value
   * @param accumulatorFunction the accumulator function
