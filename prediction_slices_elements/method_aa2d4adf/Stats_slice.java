// Source-based slice around line 589
// Method: <com.google.common.math.Stats: byte[] toByteArray()>

  /** The size of byte array representation in bytes. */
  static final int BYTES = (Long.SIZE + Double.SIZE * 4) / Byte.SIZE;

  /**
   * Gets a byte array representation of this instance.
   *
   * <p><b>Note:</b> No guarantees are made regarding stability of the representation between
   * versions.
   */
  public byte[] toByteArray() {
    ByteBuffer buff = ByteBuffer.allocate(BYTES).order(ByteOrder.LITTLE_ENDIAN);
    writeTo(buff);
    return buff.array();
  }

  /**
   * Writes to the given {@link ByteBuffer} a byte representation of this instance.
   *
   * <p><b>Note:</b> No guarantees are made regarding stability of the representation between
   * versions.
