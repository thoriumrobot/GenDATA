// Source-based slice around line 165
// Method: com.google.common.net.MediaType.MD_UTF_8

  public static final MediaType HTML_UTF_8 = createConstantUtf8(TEXT_TYPE, "html");
  public static final MediaType I_CALENDAR_UTF_8 = createConstantUtf8(TEXT_TYPE, "calendar");

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

