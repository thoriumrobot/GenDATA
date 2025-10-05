// Source-based slice around line 143
// Method: <com.google.common.io.CharSink: void writeLines(Stream,String)>

  }

  /**
   * Writes the given lines of text to this sink with each line (including the last) terminated with
   * the given line separator.
   *
   * @throws IOException if an I/O error occurs while writing to this sink
   * @since 22.0 (but only since 33.4.0 in the Android flavor)
   */
  public void writeLines(Stream<? extends CharSequence> lines, String lineSeparator)
      throws IOException {
    writeLines(lines.iterator(), lineSeparator);
  }

  private void writeLines(Iterator<? extends CharSequence> lines, String lineSeparator)
      throws IOException {
    checkNotNull(lineSeparator);

    try (Writer out = openBufferedStream()) {
      while (lines.hasNext()) {
