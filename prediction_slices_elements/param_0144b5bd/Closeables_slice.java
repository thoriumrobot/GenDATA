// Source-based slice around line 82
// Method: <com.google.common.io.Closeables: void close(Closeable,boolean)>

   * The proper capitalization would be "swallowIoException." However:
   *
   * - It might be preferable to be consistent with the JDK precedent (which they stuck with even
   *   for "UncheckedIOException").
   *
   * - If we change the name, some of our callers break because our Android Lint ParameterName check
   *   doesn't make the exception for com.google.common that internal Error Prone does: b/386402967.
   */
  @SuppressWarnings("IdentifierName")
  public static void close(@Nullable Closeable closeable, boolean swallowIOException)
      throws IOException {
    if (closeable == null) {
      return;
    }
    try {
      closeable.close();
    } catch (IOException e) {
      if (swallowIOException) {
        logger.log(Level.WARNING, "IOException thrown while closing Closeable.", e);
      } else {
