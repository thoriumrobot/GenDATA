// Source-based slice around line 80
// Method: <com.google.common.io.AppendableWriter: void write(String,int,int)>


  @Override
  public void write(String str) throws IOException {
    checkNotNull(str);
    checkNotClosed();
    target.append(str);
  }

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
