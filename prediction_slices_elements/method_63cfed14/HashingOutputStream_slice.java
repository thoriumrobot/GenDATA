// Source-based slice around line 65
// Method: <com.google.common.hash.HashingOutputStream: HashCode hash()>

  public void write(byte[] bytes, int off, int len) throws IOException {
    hasher.putBytes(bytes, off, len);
    out.write(bytes, off, len);
  }

  /**
   * Returns the {@link HashCode} based on the data written to this stream. The result is
   * unspecified if this method is called more than once on the same instance.
   */
  public HashCode hash() {
    return hasher.hash();
  }

  // Overriding close() because FilterOutputStream's close() method pre-JDK8 has bad behavior:
  // it silently ignores any exception thrown by flush(). Instead, just close the delegate stream.
  // It should flush itself if necessary.
  @Override
  public void close() throws IOException {
    out.close();
  }
