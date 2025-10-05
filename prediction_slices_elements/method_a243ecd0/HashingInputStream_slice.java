// Source-based slice around line 86
// Method: <com.google.common.hash.HashingInputStream: void mark(int)>

   * @return {@code false} always
   */
  @Override
  public boolean markSupported() {
    return false;
  }

  /** mark() is not supported for HashingInputStream */
  @Override
  public void mark(int readlimit) {}

  /**
   * reset() is not supported for HashingInputStream.
   *
   * @throws IOException this operation is not supported
   */
  @Override
  public void reset() throws IOException {
    throw new IOException("reset not supported");
  }
