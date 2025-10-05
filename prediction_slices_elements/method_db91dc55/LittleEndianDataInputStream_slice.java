// Source-based slice around line 158
// Method: <com.google.common.io.LittleEndianDataInputStream: float readFloat()>

   * Reads a {@code float} as specified by {@link DataInputStream#readFloat()}, except using
   * little-endian byte order.
   *
   * @return the next four bytes of the input stream, interpreted as a {@code float} in
   *     little-endian byte order
   * @throws IOException if an I/O error occurs
   */
  @CanIgnoreReturnValue // to skip some bytes
  @Override
  public float readFloat() throws IOException {
    return Float.intBitsToFloat(readInt());
  }

  /**
   * Reads a {@code double} as specified by {@link DataInputStream#readDouble()}, except using
   * little-endian byte order.
   *
   * @return the next eight bytes of the input stream, interpreted as a {@code double} in
   *     little-endian byte order
   * @throws IOException if an I/O error occurs
