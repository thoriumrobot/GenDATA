// Source-based slice around line 330
// Method: <com.google.common.primitives.Ints: byte[] toByteArray(int)>

  /**
   * Returns a big-endian representation of {@code value} in a 4-element byte array; equivalent to
   * {@code ByteBuffer.allocate(4).putInt(value).array()}. For example, the input value {@code
   * 0x12131415} would yield the byte array {@code {0x12, 0x13, 0x14, 0x15}}.
   *
   * <p>If you need to convert and concatenate several values (possibly even of different types),
   * use a shared {@link java.nio.ByteBuffer} instance, or use {@link
   * com.google.common.io.ByteStreams#newDataOutput()} to get a growable buffer.
   */
  public static byte[] toByteArray(int value) {
    return new byte[] {
      (byte) (value >> 24), (byte) (value >> 16), (byte) (value >> 8), (byte) value
    };
  }

  /**
   * Returns the {@code int} value whose big-endian representation is stored in the first 4 bytes of
   * {@code bytes}; equivalent to {@code ByteBuffer.wrap(bytes).getInt()}. For example, the input
   * byte array {@code {0x12, 0x13, 0x14, 0x15, 0x33}} would yield the {@code int} value {@code
   * 0x12131415}.
