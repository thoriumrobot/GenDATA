// Source-based slice around line 111
// Method: <com.google.common.primitives.UnsignedBytes: byte saturatedCast(long)>


  /**
   * Returns the {@code byte} value that, when treated as unsigned, is nearest in value to {@code
   * value}.
   *
   * @param value any {@code long} value
   * @return {@code (byte) 255} if {@code value >= 255}, {@code (byte) 0} if {@code value <= 0}, and
   *     {@code value} cast to {@code byte} otherwise
   */
  public static byte saturatedCast(long value) {
    if (value > toUnsignedInt(MAX_VALUE)) {
      return MAX_VALUE; // -1
    }
    if (value < 0) {
      return (byte) 0;
    }
    return (byte) value;
  }

  /**
