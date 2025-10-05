// Source-based slice around line 113
// Method: <com.google.common.util.concurrent.AtomicDoubleArray: void set(int,double)>

    return longBitsToDouble(longs.get(i));
  }

  /**
   * Atomically sets the element at position {@code i} to the given value.
   *
   * @param i the index
   * @param newValue the new value
   */
  public final void set(int i, double newValue) {
    long next = doubleToRawLongBits(newValue);
    longs.set(i, next);
  }

  /**
   * Eventually sets the element at position {@code i} to the given value.
   *
   * @param i the index
   * @param newValue the new value
   */
