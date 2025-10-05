// Source-based slice around line 74
// Method: <com.google.common.io.LittleEndianDataInputStream: int skipBytes(int)>

    ByteStreams.readFully(this, b);
  }

  @Override
  public void readFully(byte[] b, int off, int len) throws IOException {
    ByteStreams.readFully(this, b, off, len);
  }

  @Override
  public int skipBytes(int n) throws IOException {
    return (int) in.skip(n);
  }

  @CanIgnoreReturnValue // to skip a byte
  @Override
  public int readUnsignedByte() throws IOException {
    int b1 = in.read();
    if (b1 < 0) {
      throw new EOFException();
    }
