// Source-based slice around line 96
// Method: <com.google.common.io.AppendableWriter: void close()>

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
      ((Closeable) target).close();
    }
  }

  @Override
  public Writer append(char c) throws IOException {
    checkNotClosed();
    target.append(c);
