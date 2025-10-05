// Source-based slice around line 72
// Method: <com.google.common.hash.PrimitiveSink: PrimitiveSink putBytes(ByteBuffer)>

   * Puts the remaining bytes of a byte buffer into this sink. {@code bytes.position()} is the first
   * byte written, {@code bytes.limit() - 1} is the last. The position of the buffer will be equal
   * to the limit when this method returns.
   *
   * @param bytes a byte buffer
   * @return this instance
   * @since 23.0
   */
  @CanIgnoreReturnValue
  PrimitiveSink putBytes(ByteBuffer bytes);

  /** Puts a short into this sink. */
  @CanIgnoreReturnValue
  PrimitiveSink putShort(short s);

  /** Puts an int into this sink. */
  @CanIgnoreReturnValue
  PrimitiveSink putInt(int i);

  /** Puts a long into this sink. */
