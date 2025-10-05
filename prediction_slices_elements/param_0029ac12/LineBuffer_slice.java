// Source-based slice around line 117
// Method: <com.google.common.io.LineBuffer: void handleLine(String,String)>

  }

  /**
   * Called for each line found in the character data passed to {@link #add}.
   *
   * @param line a line of text (possibly empty), without any line separators
   * @param end the line separator; one of {@code "\r"}, {@code "\n"}, {@code "\r\n"}, or {@code ""}
   * @throws IOException if an I/O error occurs
   */
  protected abstract void handleLine(String line, String end) throws IOException;
}
