// Source-based slice around line 135
// Method: <com.google.common.io.LittleEndianDataInputStream: long readLong()>

   * Reads a {@code long} as specified by {@link DataInputStream#readLong()}, except using
   * little-endian byte order.
   *
   * @return the next eight bytes of the input stream, interpreted as a {@code long} in
   *     little-endian byte order
   * @throws IOException if an I/O error occurs
   */
  @CanIgnoreReturnValue // to skip some bytes
  @Override
  public long readLong() throws IOException {
    byte b1 = readAndCheckByte();
    byte b2 = readAndCheckByte();
    byte b3 = readAndCheckByte();
    byte b4 = readAndCheckByte();
    byte b5 = readAndCheckByte();
    byte b6 = readAndCheckByte();
    byte b7 = readAndCheckByte();
    byte b8 = readAndCheckByte();

    return Longs.fromBytes(b8, b7, b6, b5, b4, b3, b2, b1);
