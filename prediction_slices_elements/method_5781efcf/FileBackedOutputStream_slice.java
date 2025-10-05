// Source-based slice around line 224
// Method: <com.google.common.io.FileBackedOutputStream: void flush()>

    out.write(b, off, len);
  }

  @Override
  public synchronized void close() throws IOException {
    out.close();
  }

  @Override
  public synchronized void flush() throws IOException {
    out.flush();
  }

  /**
   * Checks if writing {@code len} bytes would go over threshold, and switches to file buffering if
   * so.
   */
  @GuardedBy("this")
  private void update(int len) throws IOException {
    if (memory != null && (memory.getCount() + len > fileThreshold)) {
