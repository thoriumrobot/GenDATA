// Source-based slice around line 46
// Method: <com.google.common.hash.PrimitiveSink: PrimitiveSink putBytes(byte[])>

  PrimitiveSink putByte(byte b);

  /**
   * Puts an array of bytes into this sink.
   *
   * @param bytes a byte array
   * @return this instance
   */
  @CanIgnoreReturnValue
  PrimitiveSink putBytes(byte[] bytes);

  /**
   * Puts a chunk of an array of bytes into this sink. {@code bytes[off]} is the first byte written,
   * {@code bytes[off + len - 1]} is the last.
   *
   * @param bytes a byte array
   * @param off the start offset in the array
   * @param len the number of bytes to write
   * @return this instance
   * @throws IndexOutOfBoundsException if {@code off < 0} or {@code off + len > bytes.length} or
