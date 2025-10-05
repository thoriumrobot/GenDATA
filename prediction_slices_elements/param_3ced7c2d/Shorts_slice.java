// Source-based slice around line 348
// Method: <com.google.common.primitives.Shorts: short fromBytes(byte,byte)>

  }

  /**
   * Returns the {@code short} value whose byte representation is the given 2 bytes, in big-endian
   * order; equivalent to {@code Shorts.fromByteArray(new byte[] {b1, b2})}.
   *
   * @since 7.0
   */
  @GwtIncompatible // doesn't work
  public static short fromBytes(byte b1, byte b2) {
    return (short) ((b1 << 8) | (b2 & 0xFF));
  }

  private static final class ShortConverter extends Converter<String, Short>
      implements Serializable {
    static final Converter<String, Short> INSTANCE = new ShortConverter();

    @Override
    protected Short doForward(String value) {
      return Short.decode(value);
