// Source-based slice around line 753
// Method: com.google.common.net.MediaType.ZIP

   * Extensible Resource Descriptors. This is not yet registered with the IANA, but it is specified
   * by OASIS in the <a href="http://docs.oasis-open.org/xri/xrd/v1.0/cd02/xrd-1.0-cd02.html">XRD
   * definition</a> and implemented in projects such as <a
   * href="http://code.google.com/p/webfinger/">WebFinger</a>.
   *
   * @since 14.0
   */
  public static final MediaType XRD_UTF_8 = createConstantUtf8(APPLICATION_TYPE, "xrd+xml");

  public static final MediaType ZIP = createConstant(APPLICATION_TYPE, "zip");

  /* font types */

  /**
   * A collection of font outlines as defined by <a href="https://tools.ietf.org/html/rfc8081">RFC
   * 8081</a>.
   *
   * @since 30.0
   */
  public static final MediaType FONT_COLLECTION = createConstant(FONT_TYPE, "collection");
