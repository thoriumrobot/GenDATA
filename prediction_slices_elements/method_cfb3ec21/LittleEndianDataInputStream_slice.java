// Source-based slice around line 192
// Method: <com.google.common.io.LittleEndianDataInputStream: short readShort()>

   * Reads a {@code short} as specified by {@link DataInputStream#readShort()}, except using
   * little-endian byte order.
   *
   * @return the next two bytes of the input stream, interpreted as a {@code short} in little-endian
   *     byte order.
   * @throws IOException if an I/O error occurs.
   */
  @CanIgnoreReturnValue // to skip some bytes
  @Override
  public short readShort() throws IOException {
    return (short) readUnsignedShort();
  }

  /**
   * Reads a char as specified by {@link DataInputStream#readChar()}, except using little-endian
   * byte order.
   *
   * @return the next two bytes of the input stream, interpreted as a {@code char} in little-endian
   *     byte order
   * @throws IOException if an I/O error occurs
