// Source-based slice around line 132
// Method: <com.google.common.io.CharSink: void writeLines(Stream)>


  /**
   * Writes the given lines of text to this sink with each line (including the last) terminated with
   * the operating system's default line separator. This method is equivalent to {@code
   * writeLines(lines, System.getProperty("line.separator"))}.
   *
   * @throws IOException if an I/O error occurs while writing to this sink
   * @since 22.0 (but only since 33.4.0 in the Android flavor)
   */
  public void writeLines(Stream<? extends CharSequence> lines) throws IOException {
    writeLines(lines, LINE_SEPARATOR.value());
  }

  /**
   * Writes the given lines of text to this sink with each line (including the last) terminated with
   * the given line separator.
   *
   * @throws IOException if an I/O error occurs while writing to this sink
   * @since 22.0 (but only since 33.4.0 in the Android flavor)
   */
