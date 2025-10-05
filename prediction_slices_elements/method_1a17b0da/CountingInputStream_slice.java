// Source-based slice around line 85
// Method: <com.google.common.io.CountingInputStream: void reset()>


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
    if (mark == -1) {
      throw new IOException("Mark not set");
    }

    in.reset();
    count = mark;
  }
