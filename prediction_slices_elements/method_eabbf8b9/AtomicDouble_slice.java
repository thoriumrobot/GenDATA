// Source-based slice around line 164
// Method: <com.google.common.util.concurrent.AtomicDouble: double getAndAdd(double)>

  }

  /**
   * Atomically adds the given value to the current value.
   *
   * @param delta the value to add
   * @return the previous value
   */
  @CanIgnoreReturnValue
  public final double getAndAdd(double delta) {
    return getAndAccumulate(delta, Double::sum);
  }

  /**
   * Atomically adds the given value to the current value.
   *
   * @param delta the value to add
   * @return the updated value
   */
  @CanIgnoreReturnValue
