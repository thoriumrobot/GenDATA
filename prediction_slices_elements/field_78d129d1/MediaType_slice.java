// Source-based slice around line 174
// Method: com.google.common.net.MediaType.TEXT_JAVASCRIPT_UTF_8

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
   * values</a>.
   *
   * @since 15.0
   */
  public static final MediaType TSV_UTF_8 = createConstantUtf8(TEXT_TYPE, "tab-separated-values");

  public static final MediaType VCARD_UTF_8 = createConstantUtf8(TEXT_TYPE, "vcard");
