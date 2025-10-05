// Source-based slice around line 321
// Method: <com.google.common.primitives.Shorts: byte[] toByteArray(short)>

   * Returns a big-endian representation of {@code value} in a 2-element byte array; equivalent to
   * {@code ByteBuffer.allocate(2).putShort(value).array()}. For example, the input value {@code
   * (short) 0x1234} would yield the byte array {@code {0x12, 0x34}}.
   *
   * <p>If you need to convert and concatenate several values (possibly even of different types),
   * use a shared {@link java.nio.ByteBuffer} instance, or use {@link
   * com.google.common.io.ByteStreams#newDataOutput()} to get a growable buffer.
   */
  @GwtIncompatible // doesn't work
  public static byte[] toByteArray(short value) {
    return new byte[] {(byte) (value >> 8), (byte) value};
  }

  /**
   * Returns the {@code short} value whose big-endian representation is stored in the first 2 bytes
   * of {@code bytes}; equivalent to {@code ByteBuffer.wrap(bytes).getShort()}. For example, the
   * input byte array {@code {0x54, 0x32}} would yield the {@code short} value {@code 0x5432}.
   *
   * <p>Arguably, it's preferable to use {@link java.nio.ByteBuffer}; that library exposes much more
   * flexibility at little cost in readability.
