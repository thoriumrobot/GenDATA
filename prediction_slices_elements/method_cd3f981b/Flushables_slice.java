// Source-based slice around line 52
// Method: <com.google.common.io.Flushables: void flush(Flushable,boolean)>

   *
   * @param flushable the {@code Flushable} object to be flushed.
   * @param swallowIOException if true, don't propagate IO exceptions thrown by the {@code flush}
   *     method
   * @throws IOException if {@code swallowIOException} is false and {@link Flushable#flush} throws
   *     an {@code IOException}.
   * @see Closeables#close
   */
  @SuppressWarnings("IdentifierName") // See Closeables.close
  public static void flush(Flushable flushable, boolean swallowIOException) throws IOException {
    try {
      flushable.flush();
    } catch (IOException e) {
      if (swallowIOException) {
        logger.log(Level.WARNING, "IOException thrown while flushing Flushable.", e);
      } else {
        throw e;
      }
    }
  }
