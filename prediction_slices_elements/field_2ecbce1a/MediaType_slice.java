// Source-based slice around line 535
// Method: com.google.common.net.MediaType.MANIFEST_JSON_UTF_8

   * @since 32.0.0
   */
  public static final MediaType JWT = createConstant(APPLICATION_TYPE, "jwt");

  /**
   * The <a href="http://www.w3.org/TR/appmanifest/">Manifest for a web application</a>.
   *
   * @since 19.0
   */
  public static final MediaType MANIFEST_JSON_UTF_8 =
      createConstantUtf8(APPLICATION_TYPE, "manifest+json");

  /**
   * <a href="http://www.opengeospatial.org/standards/kml/">OGC KML (Keyhole Markup Language)</a>.
   */
  public static final MediaType KML = createConstant(APPLICATION_TYPE, "vnd.google-earth.kml+xml");

  /**
   * <a href="http://www.opengeospatial.org/standards/kml/">OGC KML (Keyhole Markup Language)</a>,
   * compressed using the ZIP format into KMZ archives.
