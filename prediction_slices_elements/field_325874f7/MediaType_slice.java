// Source-based slice around line 741
// Method: com.google.common.net.MediaType.XHTML_UTF_8

  /**
   * <a href="https://tools.ietf.org/html/rfc8081">RFC 8081</a> declares {@link #FONT_WOFF2
   * font/woff2} to be the correct media type for WOFF2, but this may be necessary in certain
   * situations for compatibility.
   *
   * @since 20.0
   */
  public static final MediaType WOFF2 = createConstant(APPLICATION_TYPE, "font-woff2");

  public static final MediaType XHTML_UTF_8 = createConstantUtf8(APPLICATION_TYPE, "xhtml+xml");

  /**
   * Extensible Resource Descriptors. This is not yet registered with the IANA, but it is specified
   * by OASIS in the <a href="http://docs.oasis-open.org/xri/xrd/v1.0/cd02/xrd-1.0-cd02.html">XRD
   * definition</a> and implemented in projects such as <a
   * href="http://code.google.com/p/webfinger/">WebFinger</a>.
   *
   * @since 14.0
   */
  public static final MediaType XRD_UTF_8 = createConstantUtf8(APPLICATION_TYPE, "xrd+xml");
