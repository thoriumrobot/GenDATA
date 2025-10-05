// Source-based slice around line 47
// Method: <com.google.common.io.CountingOutputStream: long getCount()>

   * Wraps another output stream, counting the number of bytes written.
   *
   * @param out the output stream to be wrapped
   */
  public CountingOutputStream(OutputStream out) {
    super(checkNotNull(out));
  }

  /** Returns the number of bytes written. */
  public long getCount() {
    return count;
  }

  @Override
  public void write(byte[] b, int off, int len) throws IOException {
    out.write(b, off, len);
    count += len;
  }

  @Override
