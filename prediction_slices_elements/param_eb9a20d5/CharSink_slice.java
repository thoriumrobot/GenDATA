// Source-based slice around line 94
// Method: <com.google.common.io.CharSink: void write(CharSequence)>

        ? (BufferedWriter) writer
        : new BufferedWriter(writer);
  }

  /**
   * Writes the given character sequence to this sink.
   *
   * @throws IOException if an I/O error while writing to this sink
   */
  public void write(CharSequence charSequence) throws IOException {
    checkNotNull(charSequence);

    try (Writer out = openStream()) {
      out.append(charSequence);
    }
  }

  /**
   * Writes the given lines of text to this sink with each line (including the last) terminated with
   * the operating system's default line separator. This method is equivalent to {@code
