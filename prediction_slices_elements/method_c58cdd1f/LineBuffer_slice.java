// Source-based slice around line 104
// Method: <com.google.common.io.LineBuffer: void finish()>

    return sawNewline;
  }

  /**
   * Subclasses must call this method after finishing character processing, in order to ensure that
   * any unterminated line in the buffer is passed to {@link #handleLine}.
   *
   * @throws IOException if an I/O error occurs
   */
  protected void finish() throws IOException {
    if (sawReturn || line.length() > 0) {
      finishLine(false);
    }
  }

  /**
   * Called for each line found in the character data passed to {@link #add}.
   *
   * @param line a line of text (possibly empty), without any line separators
   * @param end the line separator; one of {@code "\r"}, {@code "\n"}, {@code "\r\n"}, or {@code ""}
