// Source-based slice around line 799
// Method: com.google.common.net.MediaType.FONT_WOFF


  /**
   * <a href="http://en.wikipedia.org/wiki/Web_Open_Font_Format">Web Open Font Format</a> (WOFF). <a
   * href="https://tools.ietf.org/html/rfc8081">RFC 8081</a> declares this to be the correct media
   * type for SFNT, but {@link #WOFF application/font-woff} may be necessary in certain situations
   * for compatibility.
   *
   * @since 30.0
   */
  public static final MediaType FONT_WOFF = createConstant(FONT_TYPE, "woff");

  /**
   * <a href="http://en.wikipedia.org/wiki/Web_Open_Font_Format">Web Open Font Format</a> (WOFF2).
   * <a href="https://tools.ietf.org/html/rfc8081">RFC 8081</a> declares this to be the correct
   * media type for SFNT, but {@link #WOFF2 application/font-woff2} may be necessary in certain
   * situations for compatibility.
   *
   * @since 30.0
   */
  public static final MediaType FONT_WOFF2 = createConstant(FONT_TYPE, "woff2");
