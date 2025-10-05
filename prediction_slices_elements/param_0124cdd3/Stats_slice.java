// Source-based slice around line 647
// Method: <com.google.common.math.Stats: Stats readFrom(ByteBuffer)>

   * Creates a Stats instance from the byte representation read from the given {@link ByteBuffer}.
   *
   * <p><b>Note:</b> No guarantees are made regarding stability of the representation between
   * versions.
   *
   * @param buffer A {@link ByteBuffer} with at least BYTES {@link ByteBuffer#remaining}, ordered as
   *     {@link ByteOrder#LITTLE_ENDIAN}, from which a BYTES-long byte representation of this
   *     instance is read. In the process increases the position of {@link ByteBuffer} by BYTES.
   */
  static Stats readFrom(ByteBuffer buffer) {
    checkNotNull(buffer);
    checkArgument(
        buffer.remaining() >= BYTES,
        "Expected at least Stats.BYTES = %s remaining , got %s",
        BYTES,
        buffer.remaining());
    return new Stats(
        buffer.getLong(),
        buffer.getDouble(),
        buffer.getDouble(),
