// Source-based slice around line 230
// Method: com.google.common.net.MediaType.ICO

   * {@code /etc/mime.types}, e.g. in <a href=
   * "http://anonscm.debian.org/gitweb/?p=collab-maint/mime-support.git;a=blob;f=mime.types;hb=HEAD"
   * >Debian 3.48-1</a>.
   *
   * @since 15.0
   */
  public static final MediaType CRW = createConstant(IMAGE_TYPE, "x-canon-crw");

  public static final MediaType GIF = createConstant(IMAGE_TYPE, "gif");
  public static final MediaType ICO = createConstant(IMAGE_TYPE, "vnd.microsoft.icon");
  public static final MediaType JPEG = createConstant(IMAGE_TYPE, "jpeg");
  public static final MediaType PNG = createConstant(IMAGE_TYPE, "png");

  /**
   * The Photoshop File Format ({@code psd} files) as defined by <a
   * href="http://www.iana.org/assignments/media-types/image/vnd.adobe.photoshop">IANA</a>, and
   * found in {@code /etc/mime.types}, e.g. <a
   * href="http://svn.apache.org/repos/asf/httpd/httpd/branches/1.3.x/conf/mime.types"></a> of the
   * Apache <a href="http://httpd.apache.org/">HTTPD project</a>; for the specification, see <a
   * href="http://www.adobe.com/devnet-apps/photoshop/fileformatashtml/PhotoshopFileFormats.htm">
