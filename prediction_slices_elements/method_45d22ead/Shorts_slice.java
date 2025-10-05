// Source-based slice around line 103
// Method: <com.google.common.primitives.Shorts: short saturatedCast(long)>

  }

  /**
   * Returns the {@code short} nearest in value to {@code value}.
   *
   * @param value any {@code long} value
   * @return the same value cast to {@code short} if it is in the range of the {@code short} type,
   *     {@link Short#MAX_VALUE} if it is too large, or {@link Short#MIN_VALUE} if it is too small
   */
  public static short saturatedCast(long value) {
    if (value > Short.MAX_VALUE) {
      return Short.MAX_VALUE;
    }
    if (value < Short.MIN_VALUE) {
      return Short.MIN_VALUE;
    }
    return (short) value;
  }

  /**
