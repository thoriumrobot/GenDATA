// Source-based slice around line 781
// Method: com.google.common.net.MediaType.FONT_SFNT


  /**
   * <a href="https://en.wikipedia.org/wiki/SFNT">Spline or Scalable Font Format</a> (SFNT). <a
   * href="https://tools.ietf.org/html/rfc8081">RFC 8081</a> declares this to be the correct media
   * type for SFNT, but {@link #SFNT application/font-sfnt} may be necessary in certain situations
   * for compatibility.
   *
   * @since 30.0
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
