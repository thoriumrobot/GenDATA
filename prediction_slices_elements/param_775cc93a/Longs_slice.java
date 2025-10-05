// Source-based slice around line 308
// Method: <com.google.common.primitives.Longs: long fromByteArray(byte[])>

   * of {@code bytes}; equivalent to {@code ByteBuffer.wrap(bytes).getLong()}. For example, the
   * input byte array {@code {0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19}} would yield the
   * {@code long} value {@code 0x1213141516171819L}.
   *
   * <p>Arguably, it's preferable to use {@link java.nio.ByteBuffer}; that library exposes much more
   * flexibility at little cost in readability.
   *
   * @throws IllegalArgumentException if {@code bytes} has fewer than 8 elements
   */
  public static long fromByteArray(byte[] bytes) {
    checkArgument(bytes.length >= BYTES, "array too small: %s < %s", bytes.length, BYTES);
    return fromBytes(
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]);
  }

  /**
   * Returns the {@code long} value whose byte representation is the given 8 bytes, in big-endian
   * order; equivalent to {@code Longs.fromByteArray(new byte[] {b1, b2, b3, b4, b5, b6, b7, b8})}.
   *
   * @since 7.0
