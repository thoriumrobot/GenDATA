// Source-based slice around line 289
// Method: <com.google.common.math.PairedStats: byte[] toByteArray()>

  /** The size of byte array representation in bytes. */
  private static final int BYTES = Stats.BYTES * 2 + Double.SIZE / Byte.SIZE;

  /**
   * Gets a byte array representation of this instance.
   *
   * <p><b>Note:</b> No guarantees are made regarding stability of the representation between
   * versions.
   */
  public byte[] toByteArray() {
    ByteBuffer buffer = ByteBuffer.allocate(BYTES).order(ByteOrder.LITTLE_ENDIAN);
    xStats.writeTo(buffer);
    yStats.writeTo(buffer);
    buffer.putDouble(sumOfProductsOfDeltas);
    return buffer.array();
  }

  /**
   * Creates a {@link PairedStats} instance from the given byte representation which was obtained by
   * {@link #toByteArray}.
