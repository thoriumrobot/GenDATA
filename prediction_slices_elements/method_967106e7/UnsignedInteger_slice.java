// Source-based slice around line 136
// Method: <com.google.common.primitives.UnsignedInteger: UnsignedInteger minus(UnsignedInteger)>

    return fromIntBits(this.value + checkNotNull(val).value);
  }

  /**
   * Returns the result of subtracting this and {@code val}. If the result would be negative,
   * returns the low 32 bits of the result.
   *
   * @since 14.0
   */
  public UnsignedInteger minus(UnsignedInteger val) {
    return fromIntBits(value - checkNotNull(val).value);
  }

  /**
   * Returns the result of multiplying this and {@code val}. If the result would have more than 32
   * bits, returns the low 32 bits of the result.
   *
   * @since 14.0
   */
  @J2ktIncompatible
