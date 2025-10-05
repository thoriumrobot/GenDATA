// Source-based slice around line 312
// Method: <com.google.common.primitives.Chars: byte[] toByteArray(char)>

   * Returns a big-endian representation of {@code value} in a 2-element byte array; equivalent to
   * {@code ByteBuffer.allocate(2).putChar(value).array()}. For example, the input value {@code
   * '\\u5432'} would yield the byte array {@code {0x54, 0x32}}.
   *
   * <p>If you need to convert and concatenate several values (possibly even of different types),
   * use a shared {@link java.nio.ByteBuffer} instance, or use {@link
   * com.google.common.io.ByteStreams#newDataOutput()} to get a growable buffer.
   */
  @GwtIncompatible // doesn't work
  public static byte[] toByteArray(char value) {
    return new byte[] {(byte) (value >> 8), (byte) value};
  }

  /**
   * Returns the {@code char} value whose big-endian representation is stored in the first 2 bytes
   * of {@code bytes}; equivalent to {@code ByteBuffer.wrap(bytes).getChar()}. For example, the
   * input byte array {@code {0x54, 0x32}} would yield the {@code char} value {@code '\\u5432'}.
   *
   * <p>Arguably, it's preferable to use {@link java.nio.ByteBuffer}; that library exposes much more
   * flexibility at little cost in readability.
