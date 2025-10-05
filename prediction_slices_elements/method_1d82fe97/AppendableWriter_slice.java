// Source-based slice around line 88
// Method: <com.google.common.io.AppendableWriter: void flush()>

  @Override
  public void write(String str, int off, int len) throws IOException {
    checkNotNull(str);
    checkNotClosed();
    // tricky: append takes start, end pair...
    target.append(str, off, off + len);
  }

  @Override
  public void flush() throws IOException {
    checkNotClosed();
    if (target instanceof Flushable) {
      ((Flushable) target).flush();
    }
  }

  @Override
  public void close() throws IOException {
    this.closed = true;
    if (target instanceof Closeable) {
