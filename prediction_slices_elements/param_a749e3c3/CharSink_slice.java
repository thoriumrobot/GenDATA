// Source-based slice around line 109
// Method: <com.google.common.io.CharSink: void writeLines(Iterable)>

  }

  /**
   * Writes the given lines of text to this sink with each line (including the last) terminated with
   * the operating system's default line separator. This method is equivalent to {@code
   * writeLines(lines, System.getProperty("line.separator"))}.
   *
   * @throws IOException if an I/O error occurs while writing to this sink
   */
  public void writeLines(Iterable<? extends CharSequence> lines) throws IOException {
    writeLines(lines, System.getProperty("line.separator"));
  }

  /**
   * Writes the given lines of text to this sink with each line (including the last) terminated with
   * the given line separator.
   *
   * @throws IOException if an I/O error occurs while writing to this sink
   */
  public void writeLines(Iterable<? extends CharSequence> lines, String lineSeparator)
