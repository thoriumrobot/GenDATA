// Source-based slice around line 148
// Method: <com.google.common.io.CharSink: void writeLines(Iterator,String)>

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
        out.append(lines.next()).append(lineSeparator);
      }
    }
  }

