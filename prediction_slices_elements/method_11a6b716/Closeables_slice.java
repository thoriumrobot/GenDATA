// Source-based slice around line 133
// Method: <com.google.common.io.Closeables: void closeQuietly(Reader)>

   * <p>While it's not safe in the general case to ignore exceptions that are thrown when closing an
   * I/O resource, it should generally be safe in the case of a resource that's being used only for
   * reading, such as a {@code Reader}. Unlike with writable resources, there's no chance that a
   * failure that occurs when closing the reader indicates a meaningful problem such as a failure to
   * flush all bytes to the underlying resource.
   *
   * @param reader the reader to be closed, or {@code null} in which case this method does nothing
   * @since 17.0
   */
  public static void closeQuietly(@Nullable Reader reader) {
    try {
      close(reader, true);
    } catch (IOException impossible) {
      throw new AssertionError(impossible);
    }
  }
}
