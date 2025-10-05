// Source-based slice around line 82
// Method: <com.google.common.io.CharSink: Writer openBufferedStream()>

   * required to be a {@link BufferedWriter} in order to allow implementations to simply delegate to
   * {@link #openStream()} when the stream returned by that method does not benefit from additional
   * buffering. This method returns a new, independent writer each time it is called.
   *
   * <p>The caller is responsible for ensuring that the returned writer is closed.
   *
   * @throws IOException if an I/O error occurs while opening the writer
   * @since 15.0 (in 14.0 with return type {@link BufferedWriter})
   */
  public Writer openBufferedStream() throws IOException {
    Writer writer = openStream();
    return (writer instanceof BufferedWriter)
        ? (BufferedWriter) writer
        : new BufferedWriter(writer);
  }

  /**
   * Writes the given character sequence to this sink.
   *
   * @throws IOException if an I/O error while writing to this sink
