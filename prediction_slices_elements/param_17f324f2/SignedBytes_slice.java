// Source-based slice around line 70
// Method: <com.google.common.primitives.SignedBytes: byte saturatedCast(long)>

  }

  /**
   * Returns the {@code byte} nearest in value to {@code value}.
   *
   * @param value any {@code long} value
   * @return the same value cast to {@code byte} if it is in the range of the {@code byte} type,
   *     {@link Byte#MAX_VALUE} if it is too large, or {@link Byte#MIN_VALUE} if it is too small
   */
  public static byte saturatedCast(long value) {
    if (value > Byte.MAX_VALUE) {
      return Byte.MAX_VALUE;
    }
    if (value < Byte.MIN_VALUE) {
      return Byte.MIN_VALUE;
    }
    return (byte) value;
  }

  /**
