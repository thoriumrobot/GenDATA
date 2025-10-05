// Source-based slice around line 148
// Method: <com.google.common.primitives.UnsignedInteger: UnsignedInteger times(UnsignedInteger)>


  /**
   * Returns the result of multiplying this and {@code val}. If the result would have more than 32
   * bits, returns the low 32 bits of the result.
   *
   * @since 14.0
   */
  @J2ktIncompatible
  @GwtIncompatible // Does not truncate correctly
  public UnsignedInteger times(UnsignedInteger val) {
    // TODO(lowasser): make this GWT-compatible
    return fromIntBits(value * checkNotNull(val).value);
  }

  /**
   * Returns the result of dividing this by {@code val}.
   *
   * @throws ArithmeticException if {@code val} is zero
   * @since 14.0
   */
