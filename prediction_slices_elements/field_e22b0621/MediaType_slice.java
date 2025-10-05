// Source-based slice around line 155
// Method: com.google.common.net.MediaType.CSV_UTF_8

   *
   * @since 30.0
   */
  public static final MediaType ANY_FONT_TYPE = createConstant(FONT_TYPE, WILDCARD);

  /* text types */
  public static final MediaType CACHE_MANIFEST_UTF_8 =
      createConstantUtf8(TEXT_TYPE, "cache-manifest");
  public static final MediaType CSS_UTF_8 = createConstantUtf8(TEXT_TYPE, "css");
  public static final MediaType CSV_UTF_8 = createConstantUtf8(TEXT_TYPE, "csv");
  public static final MediaType HTML_UTF_8 = createConstantUtf8(TEXT_TYPE, "html");
  public static final MediaType I_CALENDAR_UTF_8 = createConstantUtf8(TEXT_TYPE, "calendar");

  /**
   * As described in <a href="https://www.rfc-editor.org/rfc/rfc7763.html">RFC 7763</a>, this
   * constant ({@code text/markdown}) is used for Markdown documents.
   *
   * @since 33.3.0
   */
  public static final MediaType MD_UTF_8 = createConstantUtf8(TEXT_TYPE, "markdown");
