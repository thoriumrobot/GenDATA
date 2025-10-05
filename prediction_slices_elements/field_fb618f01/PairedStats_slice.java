// Source-based slice around line 281
// Method: com.google.common.math.PairedStats.BYTES

    if (value <= -1.0) {
      return -1.0;
    }
    return value;
  }

  // Serialization helpers

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
