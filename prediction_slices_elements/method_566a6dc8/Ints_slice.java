// Source-based slice around line 347
// Method: <com.google.common.primitives.Ints: int fromByteArray(byte[])>

   * {@code bytes}; equivalent to {@code ByteBuffer.wrap(bytes).getInt()}. For example, the input
   * byte array {@code {0x12, 0x13, 0x14, 0x15, 0x33}} would yield the {@code int} value {@code
   * 0x12131415}.
   *
   * <p>Arguably, it's preferable to use {@link java.nio.ByteBuffer}; that library exposes much more
   * flexibility at little cost in readability.
   *
   * @throws IllegalArgumentException if {@code bytes} has fewer than 4 elements
   */
  public static int fromByteArray(byte[] bytes) {
    checkArgument(bytes.length >= BYTES, "array too small: %s < %s", bytes.length, BYTES);
    return fromBytes(bytes[0], bytes[1], bytes[2], bytes[3]);
  }

  /**
   * Returns the {@code int} value whose byte representation is the given 4 bytes, in big-endian
   * order; equivalent to {@code Ints.fromByteArray(new byte[] {b1, b2, b3, b4})}.
   *
   * @since 7.0
   */
