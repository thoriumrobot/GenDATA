// Source-based slice around line 206
// Method: <com.google.common.io.LittleEndianDataInputStream: char readChar()>

   * Reads a char as specified by {@link DataInputStream#readChar()}, except using little-endian
   * byte order.
   *
   * @return the next two bytes of the input stream, interpreted as a {@code char} in little-endian
   *     byte order
   * @throws IOException if an I/O error occurs
   */
  @CanIgnoreReturnValue // to skip some bytes
  @Override
  public char readChar() throws IOException {
    return (char) readUnsignedShort();
  }

  @CanIgnoreReturnValue // to skip a byte
  @Override
  public byte readByte() throws IOException {
    return (byte) readUnsignedByte();
  }

  @CanIgnoreReturnValue // to skip a byte
