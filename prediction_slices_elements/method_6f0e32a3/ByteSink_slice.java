// Source-based slice around line 72
// Method: <com.google.common.io.ByteSink: OutputStream openStream()>


  /**
   * Opens a new {@link OutputStream} for writing to this sink. This method returns a new,
   * independent stream each time it is called.
   *
   * <p>The caller is responsible for ensuring that the returned stream is closed.
   *
   * @throws IOException if an I/O error occurs while opening the stream
   */
  public abstract OutputStream openStream() throws IOException;

  /**
   * Opens a new buffered {@link OutputStream} for writing to this sink. The returned stream is not
   * required to be a {@link BufferedOutputStream} in order to allow implementations to simply
   * delegate to {@link #openStream()} when the stream returned by that method does not benefit from
   * additional buffering (for example, a {@code ByteArrayOutputStream}). This method returns a new,
   * independent stream each time it is called.
   *
   * <p>The caller is responsible for ensuring that the returned stream is closed.
   *
