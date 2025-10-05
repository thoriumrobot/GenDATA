// Source-based slice around line 181
// Method: <com.google.common.util.concurrent.AtomicDoubleArray: double getAndAdd(int,double)>


  /**
   * Atomically adds the given value to the element at index {@code i}.
   *
   * @param i the index
   * @param delta the value to add
   * @return the previous value
   */
  @CanIgnoreReturnValue
  public final double getAndAdd(int i, double delta) {
    return getAndAccumulate(i, delta, Double::sum);
  }

  /**
   * Atomically adds the given value to the element at index {@code i}.
   *
   * @param i the index
   * @param delta the value to add
   * @return the updated value
   */
