// Source-based slice around line 86
// Method: <com.google.common.io.ByteSink: OutputStream openBufferedStream()>

   * delegate to {@link #openStream()} when the stream returned by that method does not benefit from
   * additional buffering (for example, a {@code ByteArrayOutputStream}). This method returns a new,
   * independent stream each time it is called.
   *
   * <p>The caller is responsible for ensuring that the returned stream is closed.
   *
   * @throws IOException if an I/O error occurs while opening the stream
   * @since 15.0 (in 14.0 with return type {@link BufferedOutputStream})
   */
  public OutputStream openBufferedStream() throws IOException {
    OutputStream out = openStream();
    return (out instanceof BufferedOutputStream)
        ? (BufferedOutputStream) out
        : new BufferedOutputStream(out);
  }

  /**
   * Writes all the given bytes to this sink.
   *
   * @throws IOException if an I/O occurs while writing to this sink
