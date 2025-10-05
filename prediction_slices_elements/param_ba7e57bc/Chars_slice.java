// Source-based slice around line 327
// Method: <com.google.common.primitives.Chars: char fromByteArray(byte[])>

   * of {@code bytes}; equivalent to {@code ByteBuffer.wrap(bytes).getChar()}. For example, the
   * input byte array {@code {0x54, 0x32}} would yield the {@code char} value {@code '\\u5432'}.
   *
   * <p>Arguably, it's preferable to use {@link java.nio.ByteBuffer}; that library exposes much more
   * flexibility at little cost in readability.
   *
   * @throws IllegalArgumentException if {@code bytes} has fewer than 2 elements
   */
  @GwtIncompatible // doesn't work
  public static char fromByteArray(byte[] bytes) {
    checkArgument(bytes.length >= BYTES, "array too small: %s < %s", bytes.length, BYTES);
    return fromBytes(bytes[0], bytes[1]);
  }

  /**
   * Returns the {@code char} value whose byte representation is the given 2 bytes, in big-endian
   * order; equivalent to {@code Chars.fromByteArray(new byte[] {b1, b2})}.
   *
   * @since 7.0
   */
