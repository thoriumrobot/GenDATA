// Source-based slice around line 181
// Method: <com.google.common.io.FileBackedOutputStream: void reset()>

    }
  }

  /**
   * Calls {@link #close} if not already closed, and then resets this object back to its initial
   * state, for reuse. If data was buffered to a file, it will be deleted.
   *
   * @throws IOException if an I/O error occurred while deleting the file buffer
   */
  public synchronized void reset() throws IOException {
    try {
      close();
    } finally {
      if (memory == null) {
        memory = new MemoryOutput();
      } else {
        memory.reset();
      }
      out = memory;
      if (file != null) {
