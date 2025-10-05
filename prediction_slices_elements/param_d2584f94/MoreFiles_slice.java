// Source-based slice around line 237
// Method: <com.google.common.io.MoreFiles: CharSource asCharSource(Path,Charset,OpenOption)>

  /**
   * Returns a view of the given {@code path} as a {@link CharSource} using the given {@code
   * charset}.
   *
   * <p>Any {@linkplain OpenOption open options} provided are used when opening streams to the file
   * and may affect the behavior of the returned source and the streams it provides. See {@link
   * StandardOpenOption} for the standard options that may be provided. Providing no options is
   * equivalent to providing the {@link StandardOpenOption#READ READ} option.
   */
  public static CharSource asCharSource(Path path, Charset charset, OpenOption... options) {
    return asByteSource(path, options).asCharSource(charset);
  }

  /**
   * Returns a view of the given {@code path} as a {@link CharSink} using the given {@code charset}.
   *
   * <p>Any {@linkplain OpenOption open options} provided are used when opening streams to the file
   * and may affect the behavior of the returned sink and the streams it provides. See {@link
   * StandardOpenOption} for the standard options that may be provided. Providing no options is
   * equivalent to providing the {@link StandardOpenOption#CREATE CREATE}, {@link
