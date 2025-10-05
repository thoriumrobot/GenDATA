// Source-based slice around line 358
// Method: <com.google.common.primitives.Ints: int fromBytes(byte,byte,byte,byte)>

    return fromBytes(bytes[0], bytes[1], bytes[2], bytes[3]);
  }

  /**
   * Returns the {@code int} value whose byte representation is the given 4 bytes, in big-endian
   * order; equivalent to {@code Ints.fromByteArray(new byte[] {b1, b2, b3, b4})}.
   *
   * @since 7.0
   */
  public static int fromBytes(byte b1, byte b2, byte b3, byte b4) {
    return b1 << 24 | (b2 & 0xFF) << 16 | (b3 & 0xFF) << 8 | (b4 & 0xFF);
  }

  private static final class IntConverter extends Converter<String, Integer>
      implements Serializable {
    static final Converter<String, Integer> INSTANCE = new IntConverter();

    @Override
    protected Integer doForward(String value) {
      return Integer.decode(value);
