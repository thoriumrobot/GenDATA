// Source-based slice around line 304
// Method: <com.google.common.math.PairedStats: PairedStats fromByteArray(byte[])>

  }

  /**
   * Creates a {@link PairedStats} instance from the given byte representation which was obtained by
   * {@link #toByteArray}.
   *
   * <p><b>Note:</b> No guarantees are made regarding stability of the representation between
   * versions.
   */
  public static PairedStats fromByteArray(byte[] byteArray) {
    checkNotNull(byteArray);
    checkArgument(
        byteArray.length == BYTES,
        "Expected PairedStats.BYTES = %s, got %s",
        BYTES,
        byteArray.length);
    ByteBuffer buffer = ByteBuffer.wrap(byteArray).order(ByteOrder.LITTLE_ENDIAN);
    Stats xStats = Stats.readFrom(buffer);
    Stats yStats = Stats.readFrom(buffer);
    double sumOfProductsOfDeltas = buffer.getDouble();
