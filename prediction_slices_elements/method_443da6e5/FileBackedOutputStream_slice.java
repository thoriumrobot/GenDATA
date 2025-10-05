// Source-based slice around line 219
// Method: <com.google.common.io.FileBackedOutputStream: void close()>

  }

  @Override
  public synchronized void write(byte[] b, int off, int len) throws IOException {
    update(len);
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
