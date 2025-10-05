// Source-based slice around line 53
// Method: <com.google.common.io.CountingInputStream: int read()>

    super(checkNotNull(in));
  }

  /** Returns the number of bytes read. */
  public long getCount() {
    return count;
  }

  @Override
  public int read() throws IOException {
    int result = in.read();
    if (result != -1) {
      count++;
    }
    return result;
  }

  @Override
  public int read(byte[] b, int off, int len) throws IOException {
    int result = in.read(b, off, len);
