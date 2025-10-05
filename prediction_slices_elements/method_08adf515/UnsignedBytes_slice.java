// Source-based slice around line 85
// Method: <com.google.common.primitives.UnsignedBytes: int toInt(byte)>

  /**
   * Returns the value of the given byte as an integer, when treated as unsigned. That is, returns
   * {@code value + 256} if {@code value} is negative; {@code value} itself otherwise.
   *
   * <p>Prefer {@link Byte#toUnsignedInt(byte)} instead.
   *
   * @since 6.0
   */
  @InlineMe(replacement = "Byte.toUnsignedInt(value)")
  public static int toInt(byte value) {
    return Byte.toUnsignedInt(value);
  }

  /**
   * Returns the {@code byte} value that, when treated as unsigned, is equal to {@code value}, if
   * possible.
   *
   * @param value a value between 0 and 255 inclusive
   * @return the {@code byte} value that, when treated as unsigned, equals {@code value}
   * @throws IllegalArgumentException if {@code value} is negative or greater than 255
