// Source-based slice around line 103
// Method: <com.google.common.util.concurrent.AtomicDoubleArray: double get(int)>

    return longs.length();
  }

  /**
   * Gets the current value at position {@code i}.
   *
   * @param i the index
   * @return the current value
   */
  public final double get(int i) {
    return longBitsToDouble(longs.get(i));
  }

  /**
   * Atomically sets the element at position {@code i} to the given value.
   *
   * @param i the index
   * @param newValue the new value
   */
  public final void set(int i, double newValue) {
