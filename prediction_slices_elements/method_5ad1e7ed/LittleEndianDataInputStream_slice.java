// Source-based slice around line 172
// Method: <com.google.common.io.LittleEndianDataInputStream: double readDouble()>

   * Reads a {@code double} as specified by {@link DataInputStream#readDouble()}, except using
   * little-endian byte order.
   *
   * @return the next eight bytes of the input stream, interpreted as a {@code double} in
   *     little-endian byte order
   * @throws IOException if an I/O error occurs
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
