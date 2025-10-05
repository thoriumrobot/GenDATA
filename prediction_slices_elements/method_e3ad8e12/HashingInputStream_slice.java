// Source-based slice around line 102
// Method: <com.google.common.hash.HashingInputStream: HashCode hash()>

  @Override
  public void reset() throws IOException {
    throw new IOException("reset not supported");
  }

  /**
   * Returns the {@link HashCode} based on the data read from this stream. The result is unspecified
   * if this method is called more than once on the same instance.
   */
  public HashCode hash() {
    return hasher.hash();
  }
}
