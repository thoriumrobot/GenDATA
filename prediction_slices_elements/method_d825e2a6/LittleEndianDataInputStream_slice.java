// Source-based slice around line 212
// Method: <com.google.common.io.LittleEndianDataInputStream: byte readByte()>

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
  @Override
  public boolean readBoolean() throws IOException {
    return readUnsignedByte() != 0;
  }

  /**
