// Source-based slice around line 116
// Method: <com.google.common.io.LittleEndianDataInputStream: int readInt()>

   * Reads an integer as specified by {@link DataInputStream#readInt()}, except using little-endian
   * byte order.
   *
   * @return the next four bytes of the input stream, interpreted as an {@code int} in little-endian
   *     byte order
   * @throws IOException if an I/O error occurs
   */
  @CanIgnoreReturnValue // to skip some bytes
  @Override
  public int readInt() throws IOException {
    byte b1 = readAndCheckByte();
    byte b2 = readAndCheckByte();
    byte b3 = readAndCheckByte();
    byte b4 = readAndCheckByte();

    return Ints.fromBytes(b4, b3, b2, b1);
  }

  /**
   * Reads a {@code long} as specified by {@link DataInputStream#readLong()}, except using
