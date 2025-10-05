// Source-based slice around line 84
// Method: <com.google.common.io.MoreFiles: ByteSource asByteSource(Path,OpenOption)>


  /**
   * Returns a view of the given {@code path} as a {@link ByteSource}.
   *
   * <p>Any {@linkplain OpenOption open options} provided are used when opening streams to the file
   * and may affect the behavior of the returned source and the streams it provides. See {@link
   * StandardOpenOption} for the standard options that may be provided. Providing no options is
   * equivalent to providing the {@link StandardOpenOption#READ READ} option.
   */
  public static ByteSource asByteSource(Path path, OpenOption... options) {
    return new PathByteSource(path, options);
  }

  private static final class PathByteSource extends
      ByteSource
  {

    private static final LinkOption[] FOLLOW_LINKS = {};

    private final Path path;
