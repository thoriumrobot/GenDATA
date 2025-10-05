// Source-based slice around line 58
// Method: <com.google.common.io.CountingOutputStream: void write(int)>

  }

  @Override
  public void write(byte[] b, int off, int len) throws IOException {
    out.write(b, off, len);
    count += len;
  }

  @Override
  public void write(int b) throws IOException {
    out.write(b);
    count++;
  }

  // Overriding close() because FilterOutputStream's close() method pre-JDK8 has bad behavior:
  // it silently ignores any exception thrown by flush(). Instead, just close the delegate stream.
  // It should flush itself if necessary.
  @Override
  public void close() throws IOException {
    out.close();
