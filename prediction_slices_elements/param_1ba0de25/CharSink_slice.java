// Source-based slice around line 168
// Method: <com.google.common.io.CharSink: long writeFrom(Readable)>

  /**
   * Writes all the text from the given {@link Readable} (such as a {@link Reader}) to this sink.
   * Does not close {@code readable} if it is {@code Closeable}.
   *
   * @return the number of characters written
   * @throws IOException if an I/O error occurs while reading from {@code readable} or writing to
   *     this sink
   */
  @CanIgnoreReturnValue
  public long writeFrom(Readable readable) throws IOException {
    checkNotNull(readable);

    try (Writer out = openStream()) {
      return CharStreams.copy(readable, out);
    }
  }
}
