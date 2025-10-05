// Source-based slice around line 218
// Method: <com.google.common.io.LittleEndianDataInputStream: boolean readBoolean()>


  @CanIgnoreReturnValue // to skip a byte
  @Override
  public byte readByte() throws IOException {
    return (byte) readUnsignedByte();
  }

  @CanIgnoreReturnValue // to skip a byte
  @Override
  public boolean readBoolean() throws IOException {
    return readUnsignedByte() != 0;
  }

  /**
   * Reads a byte from the input stream checking that the end of file (EOF) has not been
   * encountered.
   *
   * @return byte read from input
   * @throws IOException if an error is encountered while reading
   * @throws EOFException if the end of file (EOF) is encountered.
