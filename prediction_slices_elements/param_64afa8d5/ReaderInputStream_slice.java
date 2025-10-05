// Source-based slice around line 130
// Method: <com.google.common.io.ReaderInputStream: int read(byte[],int,int)>


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
      return 0;
    }

    // The rest of this method implements the process described by the CharsetEncoder javadoc.
    int totalBytesRead = 0;
    boolean doneEncoding = endOfInput;

