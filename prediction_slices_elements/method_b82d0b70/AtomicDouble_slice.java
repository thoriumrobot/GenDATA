// Source-based slice around line 282
// Method: <com.google.common.util.concurrent.AtomicDouble: float floatValue()>

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
  }

  /** Returns the value of this {@code AtomicDouble} as a {@code double}. */
  @Override
  public double doubleValue() {
    return get();
  }

  /**
