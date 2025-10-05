// Source-based slice around line 76
// Method: <com.google.common.hash.PrimitiveSink: PrimitiveSink putShort(short)>

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
  @CanIgnoreReturnValue
  PrimitiveSink putLong(long l);

  /** Puts a float into this sink. */
