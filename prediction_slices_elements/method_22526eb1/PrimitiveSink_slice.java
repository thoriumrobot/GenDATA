// Source-based slice around line 60
// Method: <com.google.common.hash.PrimitiveSink: PrimitiveSink putBytes(byte[],int,int)>

   *
   * @param bytes a byte array
   * @param off the start offset in the array
   * @param len the number of bytes to write
   * @return this instance
   * @throws IndexOutOfBoundsException if {@code off < 0} or {@code off + len > bytes.length} or
   *     {@code len < 0}
   */
  @CanIgnoreReturnValue
  PrimitiveSink putBytes(byte[] bytes, int off, int len);

  /**
   * Puts the remaining bytes of a byte buffer into this sink. {@code bytes.position()} is the first
   * byte written, {@code bytes.limit() - 1} is the last. The position of the buffer will be equal
   * to the limit when this method returns.
   *
   * @param bytes a byte buffer
   * @return this instance
   * @since 23.0
   */
