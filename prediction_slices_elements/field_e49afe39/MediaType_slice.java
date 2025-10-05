// Source-based slice around line 789
// Method: com.google.common.net.MediaType.FONT_TTF

   */
  public static final MediaType FONT_SFNT = createConstant(FONT_TYPE, "sfnt");

  /**
   * <a href="https://en.wikipedia.org/wiki/TrueType">True Type Font Format</a> (TTF) as defined by
   * <a href="https://tools.ietf.org/html/rfc8081">RFC 8081</a>.
   *
   * @since 30.0
   */
  public static final MediaType FONT_TTF = createConstant(FONT_TYPE, "ttf");

  /**
   * <a href="http://en.wikipedia.org/wiki/Web_Open_Font_Format">Web Open Font Format</a> (WOFF). <a
   * href="https://tools.ietf.org/html/rfc8081">RFC 8081</a> declares this to be the correct media
   * type for SFNT, but {@link #WOFF application/font-woff} may be necessary in certain situations
   * for compatibility.
   *
   * @since 30.0
   */
  public static final MediaType FONT_WOFF = createConstant(FONT_TYPE, "woff");
