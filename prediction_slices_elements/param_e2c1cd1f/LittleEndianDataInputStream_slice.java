// Source-based slice around line 69
// Method: <com.google.common.io.LittleEndianDataInputStream: void readFully(byte[],int,int)>

    throw new UnsupportedOperationException("readLine is not supported");
  }

  @Override
  public void readFully(byte[] b) throws IOException {
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
