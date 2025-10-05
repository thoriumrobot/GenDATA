// Source-based slice around line 771
// Method: com.google.common.net.MediaType.FONT_OTF

   */
  public static final MediaType FONT_COLLECTION = createConstant(FONT_TYPE, "collection");

  /**
   * <a href="https://en.wikipedia.org/wiki/OpenType">Open Type Font Format</a> (OTF) as defined by
   * <a href="https://tools.ietf.org/html/rfc8081">RFC 8081</a>.
   *
   * @since 30.0
   */
  public static final MediaType FONT_OTF = createConstant(FONT_TYPE, "otf");

  /**
   * <a href="https://en.wikipedia.org/wiki/SFNT">Spline or Scalable Font Format</a> (SFNT). <a
   * href="https://tools.ietf.org/html/rfc8081">RFC 8081</a> declares this to be the correct media
   * type for SFNT, but {@link #SFNT application/font-sfnt} may be necessary in certain situations
   * for compatibility.
   *
   * @since 30.0
   */
  public static final MediaType FONT_SFNT = createConstant(FONT_TYPE, "sfnt");
