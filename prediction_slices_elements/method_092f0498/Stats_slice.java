// Source-based slice around line 605
// Method: <com.google.common.math.Stats: void writeTo(ByteBuffer)>

   * Writes to the given {@link ByteBuffer} a byte representation of this instance.
   *
   * <p><b>Note:</b> No guarantees are made regarding stability of the representation between
   * versions.
   *
   * @param buffer A {@link ByteBuffer} with at least BYTES {@link ByteBuffer#remaining}, ordered as
   *     {@link ByteOrder#LITTLE_ENDIAN}, to which a BYTES-long byte representation of this instance
   *     is written. In the process increases the position of {@link ByteBuffer} by BYTES.
   */
  void writeTo(ByteBuffer buffer) {
    checkNotNull(buffer);
    checkArgument(
        buffer.remaining() >= BYTES,
        "Expected at least Stats.BYTES = %s remaining , got %s",
        BYTES,
        buffer.remaining());
    buffer
        .putLong(count)
        .putDouble(mean)
        .putDouble(sumOfSquaresOfDeltas)
