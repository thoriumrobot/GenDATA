// Source-based slice around line 697
// Method: com.google.common.net.MediaType.SHOCKWAVE_FLASH

  /**
   * <a href="https://tools.ietf.org/html/rfc8081">RFC 8081</a> declares {@link #FONT_SFNT
   * font/sfnt} to be the correct media type for SFNT, but this may be necessary in certain
   * situations for compatibility.
   *
   * @since 17.0
   */
  public static final MediaType SFNT = createConstant(APPLICATION_TYPE, "font-sfnt");

  public static final MediaType SHOCKWAVE_FLASH =
      createConstant(APPLICATION_TYPE, "x-shockwave-flash");

  /**
   * {@code skp} files produced by the 3D Modeling software <a
   * href="https://www.sketchup.com/">SketchUp</a>
   *
   * @since 13.0
   */
  public static final MediaType SKETCHUP = createConstant(APPLICATION_TYPE, "vnd.sketchup.skp");

