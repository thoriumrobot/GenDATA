// Source-based slice around line 730
// Method: com.google.common.net.MediaType.WOFF

  public static final MediaType TAR = createConstant(APPLICATION_TYPE, "x-tar");

  /**
   * <a href="https://tools.ietf.org/html/rfc8081">RFC 8081</a> declares {@link #FONT_WOFF
   * font/woff} to be the correct media type for WOFF, but this may be necessary in certain
   * situations for compatibility.
   *
   * @since 17.0
   */
  public static final MediaType WOFF = createConstant(APPLICATION_TYPE, "font-woff");

  /**
   * <a href="https://tools.ietf.org/html/rfc8081">RFC 8081</a> declares {@link #FONT_WOFF2
   * font/woff2} to be the correct media type for WOFF2, but this may be necessary in certain
   * situations for compatibility.
   *
   * @since 20.0
   */
  public static final MediaType WOFF2 = createConstant(APPLICATION_TYPE, "font-woff2");

