// Source-based slice around line 99
// Method: <com.google.common.primitives.Chars: char saturatedCast(long)>


  /**
   * Returns the {@code char} nearest in value to {@code value}.
   *
   * @param value any {@code long} value
   * @return the same value cast to {@code char} if it is in the range of the {@code char} type,
   *     {@link Character#MAX_VALUE} if it is too large, or {@link Character#MIN_VALUE} if it is too
   *     small
   */
  public static char saturatedCast(long value) {
    if (value > Character.MAX_VALUE) {
      return Character.MAX_VALUE;
    }
    if (value < Character.MIN_VALUE) {
      return Character.MIN_VALUE;
    }
    return (char) value;
  }

  /**
