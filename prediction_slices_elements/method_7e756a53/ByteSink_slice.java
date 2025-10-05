// Source-based slice around line 114
// Method: <com.google.common.io.ByteSink: long writeFrom(InputStream)>


  /**
   * Writes all the bytes from the given {@code InputStream} to this sink. Does not close {@code
   * input}.
   *
   * @return the number of bytes written
   * @throws IOException if an I/O occurs while reading from {@code input} or writing to this sink
   */
  @CanIgnoreReturnValue
  public long writeFrom(InputStream input) throws IOException {
    checkNotNull(input);

    try (OutputStream out = openStream()) {
      return ByteStreams.copy(input, out);
    }
  }

  /**
   * A char sink that encodes written characters with a charset and writes resulting bytes to this
   * byte sink.
