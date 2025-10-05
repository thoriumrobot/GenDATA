// Source-based slice around line 196
// Method: <com.google.common.primitives.UnsignedInteger: float floatValue()>

  public long longValue() {
    return toLong(value);
  }

  /**
   * Returns the value of this {@code UnsignedInteger} as a {@code float}, analogous to a widening
   * primitive conversion from {@code int} to {@code float}, and correctly rounded.
   */
  @Override
  public float floatValue() {
    return longValue();
  }

  /**
   * Returns the value of this {@code UnsignedInteger} as a {@code double}, analogous to a widening
   * primitive conversion from {@code int} to {@code double}, and correctly rounded.
   */
  @Override
  public double doubleValue() {
    return longValue();
