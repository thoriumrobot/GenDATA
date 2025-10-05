// Source-based slice around line 202
// Method: <com.google.common.io.MoreFiles: ByteSink asByteSink(Path,OpenOption)>

   * Returns a view of the given {@code path} as a {@link ByteSink}.
   *
   * <p>Any {@linkplain OpenOption open options} provided are used when opening streams to the file
   * and may affect the behavior of the returned sink and the streams it provides. See {@link
   * StandardOpenOption} for the standard options that may be provided. Providing no options is
   * equivalent to providing the {@link StandardOpenOption#CREATE CREATE}, {@link
   * StandardOpenOption#TRUNCATE_EXISTING TRUNCATE_EXISTING} and {@link StandardOpenOption#WRITE
   * WRITE} options.
   */
  public static ByteSink asByteSink(Path path, OpenOption... options) {
    return new PathByteSink(path, options);
  }

  private static final class PathByteSink extends ByteSink {

    private final Path path;
    private final OpenOption[] options;

    private PathByteSink(Path path, OpenOption... options) {
      this.path = checkNotNull(path);
