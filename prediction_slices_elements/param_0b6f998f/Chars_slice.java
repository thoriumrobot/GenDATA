// Source-based slice around line 85
// Method: <com.google.common.primitives.Chars: char checkedCast(long)>


  /**
   * Returns the {@code char} value that is equal to {@code value}, if possible.
   *
   * @param value any value in the range of the {@code char} type
   * @return the {@code char} value that equals {@code value}
   * @throws IllegalArgumentException if {@code value} is greater than {@link Character#MAX_VALUE}
   *     or less than {@link Character#MIN_VALUE}
   */
  public static char checkedCast(long value) {
    char result = (char) value;
    checkArgument(result == value, "Out of range: %s", value);
    return result;
  }

  /**
   * Returns the {@code char} nearest in value to {@code value}.
   *
   * @param value any {@code long} value
   * @return the same value cast to {@code char} if it is in the range of the {@code char} type,
