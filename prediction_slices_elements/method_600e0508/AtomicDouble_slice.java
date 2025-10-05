// Source-based slice around line 91
// Method: <com.google.common.util.concurrent.AtomicDouble: double get()>

  public AtomicDouble() {
    // assert doubleToRawLongBits(0.0) == 0L;
  }

  /**
   * Gets the current value.
   *
   * @return the current value
   */
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
