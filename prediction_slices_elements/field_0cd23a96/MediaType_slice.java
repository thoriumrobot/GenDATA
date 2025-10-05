// Source-based slice around line 167
// Method: com.google.common.net.MediaType.PLAIN_TEXT_UTF_8


  /**
   * As described in <a href="https://www.rfc-editor.org/rfc/rfc7763.html">RFC 7763</a>, this
   * constant ({@code text/markdown}) is used for Markdown documents.
   *
   * @since 33.3.0
   */
  public static final MediaType MD_UTF_8 = createConstantUtf8(TEXT_TYPE, "markdown");

  public static final MediaType PLAIN_TEXT_UTF_8 = createConstantUtf8(TEXT_TYPE, "plain");

  /**
   * <a href="http://www.rfc-editor.org/rfc/rfc4329.txt">RFC 4329</a> declares {@link
   * #JAVASCRIPT_UTF_8 application/javascript} to be the correct media type for JavaScript, but this
   * may be necessary in certain situations for compatibility.
   */
  public static final MediaType TEXT_JAVASCRIPT_UTF_8 = createConstantUtf8(TEXT_TYPE, "javascript");

  /**
   * <a href="http://www.iana.org/assignments/media-types/text/tab-separated-values">Tab separated
