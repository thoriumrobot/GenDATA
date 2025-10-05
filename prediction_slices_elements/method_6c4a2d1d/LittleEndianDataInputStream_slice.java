// Source-based slice around line 99
// Method: <com.google.common.io.LittleEndianDataInputStream: int readUnsignedShort()>

   * Reads an unsigned {@code short} as specified by {@link DataInputStream#readUnsignedShort()},
   * except using little-endian byte order.
   *
   * @return the next two bytes of the input stream, interpreted as an unsigned 16-bit integer in
   *     little-endian byte order
   * @throws IOException if an I/O error occurs
   */
  @CanIgnoreReturnValue // to skip some bytes
  @Override
  public int readUnsignedShort() throws IOException {
    byte b1 = readAndCheckByte();
    byte b2 = readAndCheckByte();

    return Ints.fromBytes((byte) 0, (byte) 0, b2, b1);
  }

  /**
   * Reads an integer as specified by {@link DataInputStream#readInt()}, except using little-endian
   * byte order.
   *
