// Source-based slice around line 48
// Method: <com.google.common.io.CountingInputStream: long getCount()>

   * Wraps another input stream, counting the number of bytes read.
   *
   * @param in the input stream to be wrapped
   */
  public CountingInputStream(InputStream in) {
    super(checkNotNull(in));
  }

  /** Returns the number of bytes read. */
  public long getCount() {
    return count;
  }

  @Override
  public int read() throws IOException {
    int result = in.read();
    if (result != -1) {
      count++;
    }
    return result;
