// Source-based slice around line 116
// Method: <com.google.common.primitives.UnsignedInteger: UnsignedInteger valueOf(String,int)>

  }

  /**
   * Returns an {@code UnsignedInteger} holding the value of the specified {@code String}, parsed as
   * an unsigned {@code int} value in the specified radix.
   *
   * @throws NumberFormatException if the string does not contain a parsable unsigned {@code int}
   *     value
   */
  public static UnsignedInteger valueOf(String string, int radix) {
    return fromIntBits(UnsignedInts.parseUnsignedInt(string, radix));
  }

  /**
   * Returns the result of adding this and {@code val}. If the result would have more than 32 bits,
   * returns the low 32 bits of the result.
   *
   * @since 14.0
   */
  public UnsignedInteger plus(UnsignedInteger val) {
