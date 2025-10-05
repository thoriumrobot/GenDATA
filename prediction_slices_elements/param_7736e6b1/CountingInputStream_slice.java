// Source-based slice around line 78
// Method: <com.google.common.io.CountingInputStream: void mark(int)>


  @Override
  public long skip(long n) throws IOException {
    long result = in.skip(n);
    count += result;
    return result;
  }

  @Override
  public synchronized void mark(int readlimit) {
    in.mark(readlimit);
    mark = count;
    // it's okay to mark even if mark isn't supported, as reset won't work
  }

  @Override
  public synchronized void reset() throws IOException {
    if (!in.markSupported()) {
      throw new IOException("Mark not supported");
    }
