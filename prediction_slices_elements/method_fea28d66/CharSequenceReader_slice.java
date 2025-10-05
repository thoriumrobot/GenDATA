// Source-based slice around line 100
// Method: <com.google.common.io.CharSequenceReader: int read(char[],int,int)>


  @Override
  public synchronized int read() throws IOException {
    checkOpen();
    requireNonNull(seq); // safe because of checkOpen
    return hasRemaining() ? seq.charAt(pos++) : -1;
  }

  @Override
  public synchronized int read(char[] cbuf, int off, int len) throws IOException {
    checkPositionIndexes(off, off + len, cbuf.length);
    checkOpen();
    requireNonNull(seq); // safe because of checkOpen
    if (!hasRemaining()) {
      return -1;
    }
    int charsToRead = min(len, remaining());
    for (int i = 0; i < charsToRead; i++) {
      cbuf[off + i] = seq.charAt(pos++);
    }
