// Source-based slice around line 100
// Method: <com.google.common.util.concurrent.AtomicDouble: void set(double)>

  public final double get() {
    return longBitsToDouble(value);
  }

  /**
   * Sets to the given value.
   *
   * @param newValue the new value
   */
  public final void set(double newValue) {
    long next = doubleToRawLongBits(newValue);
    value = next;
  }

  /**
   * Eventually sets to the given value.
   *
   * @param newValue the new value
   */
  public final void lazySet(double newValue) {
