// Source-based slice around line 136
// Method: <com.google.common.util.concurrent.AtomicDoubleArray: double getAndSet(int,double)>

  }

  /**
   * Atomically sets the element at position {@code i} to the given value and returns the old value.
   *
   * @param i the index
   * @param newValue the new value
   * @return the previous value
   */
  public final double getAndSet(int i, double newValue) {
    long next = doubleToRawLongBits(newValue);
    return longBitsToDouble(longs.getAndSet(i, next));
  }

  /**
   * Atomically sets the element at position {@code i} to the given updated value if the current
   * value is <a href="#bitEquals">bitwise equal</a> to the expected value.
   *
   * @param i the index
   * @param expect the expected value
