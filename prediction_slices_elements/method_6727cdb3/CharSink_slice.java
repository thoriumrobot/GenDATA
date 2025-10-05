// Source-based slice around line 69
// Method: <com.google.common.io.CharSink: Writer openStream()>


  /**
   * Opens a new {@link Writer} for writing to this sink. This method returns a new, independent
   * writer each time it is called.
   *
   * <p>The caller is responsible for ensuring that the returned writer is closed.
   *
   * @throws IOException if an I/O error occurs while opening the writer
   */
  public abstract Writer openStream() throws IOException;

  /**
   * Opens a new buffered {@link Writer} for writing to this sink. The returned stream is not
   * required to be a {@link BufferedWriter} in order to allow implementations to simply delegate to
   * {@link #openStream()} when the stream returned by that method does not benefit from additional
   * buffering. This method returns a new, independent writer each time it is called.
   *
   * <p>The caller is responsible for ensuring that the returned writer is closed.
   *
   * @throws IOException if an I/O error occurs while opening the writer
