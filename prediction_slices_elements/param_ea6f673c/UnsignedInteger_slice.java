// Source-based slice around line 126
// Method: <com.google.common.primitives.UnsignedInteger: UnsignedInteger plus(UnsignedInteger)>

    return fromIntBits(UnsignedInts.parseUnsignedInt(string, radix));
  }

  /**
   * Returns the result of adding this and {@code val}. If the result would have more than 32 bits,
   * returns the low 32 bits of the result.
   *
   * @since 14.0
   */
  public UnsignedInteger plus(UnsignedInteger val) {
    return fromIntBits(this.value + checkNotNull(val).value);
  }

  /**
   * Returns the result of subtracting this and {@code val}. If the result would be negative,
   * returns the low 32 bits of the result.
   *
   * @since 14.0
   */
  public UnsignedInteger minus(UnsignedInteger val) {
