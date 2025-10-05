// Source-based slice around line 178
// Method: <com.google.common.io.LittleEndianDataInputStream: String readUTF()>

   */
  @CanIgnoreReturnValue // to skip some bytes
  @Override
  public double readDouble() throws IOException {
    return Double.longBitsToDouble(readLong());
  }

  @CanIgnoreReturnValue // to skip a field
  @Override
  public String readUTF() throws IOException {
    return new DataInputStream(in).readUTF();
  }

  /**
   * Reads a {@code short} as specified by {@link DataInputStream#readShort()}, except using
   * little-endian byte order.
   *
   * @return the next two bytes of the input stream, interpreted as a {@code short} in little-endian
   *     byte order.
   * @throws IOException if an I/O error occurs.
