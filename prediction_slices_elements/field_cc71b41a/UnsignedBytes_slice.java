// Source-based slice around line 72
// Method: com.google.common.primitives.UnsignedBytes.MAX_VALUE

   * @since 10.0
   */
  public static final byte MAX_POWER_OF_TWO = (byte) 0x80;

  /**
   * The largest value that fits into an unsigned byte.
   *
   * @since 13.0
   */
  public static final byte MAX_VALUE = (byte) 0xFF;

  private static final int UNSIGNED_MASK = 0xFF;

  /**
   * Returns the value of the given byte as an integer, when treated as unsigned. That is, returns
   * {@code value + 256} if {@code value} is negative; {@code value} itself otherwise.
   *
   * <p>Prefer {@link Byte#toUnsignedInt(byte)} instead.
   *
   * @since 6.0
