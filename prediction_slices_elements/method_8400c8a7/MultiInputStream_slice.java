// Source-based slice around line 51
// Method: <com.google.common.io.MultiInputStream: void close()>

   *
   * @param it an iterator of I/O suppliers that will provide each substream
   */
  public MultiInputStream(Iterator<? extends ByteSource> it) throws IOException {
    this.it = checkNotNull(it);
    advance();
  }

  @Override
  public void close() throws IOException {
    if (in != null) {
      try {
        in.close();
      } finally {
        in = null;
      }
    }
  }

  /** Closes the current input stream and opens the next one, if any. */
