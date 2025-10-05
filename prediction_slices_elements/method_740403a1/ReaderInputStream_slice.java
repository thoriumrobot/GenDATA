// Source-based slice around line 123
// Method: <com.google.common.io.ReaderInputStream: int read()>

    byteBuffer = ByteBuffer.allocate(bufferSize);
  }

  @Override
  public void close() throws IOException {
    reader.close();
  }

  @Override
  public int read() throws IOException {
    return (read(singleByte) == 1) ? toUnsignedInt(singleByte[0]) : -1;
  }

  // TODO(chrisn): Consider trying to encode/flush directly to the argument byte
  // buffer when possible.
  @Override
  public int read(byte[] b, int off, int len) throws IOException {
    // Obey InputStream contract.
    checkPositionIndexes(off, off + len, b.length);
    if (len == 0) {
