// Source-based slice around line 273
// Method: <com.google.common.util.concurrent.AtomicDouble: long longValue()>

  public int intValue() {
    return (int) get();
  }

  /**
   * Returns the value of this {@code AtomicDouble} as a {@code long} after a narrowing primitive
   * conversion.
   */
  @Override
  public long longValue() {
    return (long) get();
  }

  /**
   * Returns the value of this {@code AtomicDouble} as a {@code float} after a narrowing primitive
   * conversion.
   */
  @Override
  public float floatValue() {
    return (float) get();
