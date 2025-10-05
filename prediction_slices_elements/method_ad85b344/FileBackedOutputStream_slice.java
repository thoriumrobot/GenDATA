// Source-based slice around line 233
// Method: <com.google.common.io.FileBackedOutputStream: void update(int)>

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
      File temp = TempFileCreator.INSTANCE.createTempFile("FileBackedOutputStream");
      if (resetOnFinalize) {
        // Finalizers are not guaranteed to be called on system shutdown;
        // this is insurance.
        temp.deleteOnExit();
      }
      try {
        FileOutputStream transfer = new FileOutputStream(temp);
        transfer.write(memory.getBuffer(), 0, memory.getCount());
