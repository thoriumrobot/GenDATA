// Source-based slice around line 112
// Method: <com.google.common.io.Closeables: void closeQuietly(InputStream)>

   * I/O resource, it should generally be safe in the case of a resource that's being used only for
   * reading, such as an {@code InputStream}. Unlike with writable resources, there's no chance that
   * a failure that occurs when closing the stream indicates a meaningful problem such as a failure
   * to flush all bytes to the underlying resource.
   *
   * @param inputStream the input stream to be closed, or {@code null} in which case this method
   *     does nothing
   * @since 17.0
   */
  public static void closeQuietly(@Nullable InputStream inputStream) {
    try {
      close(inputStream, true);
    } catch (IOException impossible) {
      throw new AssertionError(impossible);
    }
  }

  /**
   * Closes the given {@link Reader}, logging any {@code IOException} that's thrown rather than
   * propagating it.
