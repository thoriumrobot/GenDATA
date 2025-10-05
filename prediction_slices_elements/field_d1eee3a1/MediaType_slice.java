// Source-based slice around line 541
// Method: com.google.common.net.MediaType.KML

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
   */
  public static final MediaType KMZ = createConstant(APPLICATION_TYPE, "vnd.google-earth.kmz");

  /**
   * The <a href="https://tools.ietf.org/html/rfc4155">mbox database format</a>.
   *
