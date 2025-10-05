// Source-based slice around line 73
// Method: <com.google.common.hash.HashingOutputStream: void close()>

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
}
