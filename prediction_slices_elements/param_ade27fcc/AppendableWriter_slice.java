// Source-based slice around line 73
// Method: <com.google.common.io.AppendableWriter: void write(String)>

   */

  @Override
  public void write(int c) throws IOException {
    checkNotClosed();
    target.append((char) c);
  }

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
