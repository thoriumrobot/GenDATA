// Source-based slice around line 264
// Method: <com.google.common.util.concurrent.AtomicDouble: int intValue()>

  public String toString() {
    return Double.toString(get());
  }

  /**
   * Returns the value of this {@code AtomicDouble} as an {@code int} after a narrowing primitive
   * conversion.
   */
  @Override
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
