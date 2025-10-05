// Source-based slice around line 633
// Method: com.google.common.net.MediaType.NACL_PORTABLE_APPLICATION

  public static final MediaType NACL_APPLICATION = createConstant(APPLICATION_TYPE, "x-nacl");

  /**
   * NaCl portable applications. For more information see <a
   * href="https://developer.chrome.com/native-client/devguide/coding/application-structure">the
   * Developer Guide for Native Client Application Structure</a>.
   *
   * @since 20.0
   */
  public static final MediaType NACL_PORTABLE_APPLICATION =
      createConstant(APPLICATION_TYPE, "x-pnacl");

  public static final MediaType OCTET_STREAM = createConstant(APPLICATION_TYPE, "octet-stream");

  public static final MediaType OGG_CONTAINER = createConstant(APPLICATION_TYPE, "ogg");
  public static final MediaType OOXML_DOCUMENT =
      createConstant(
          APPLICATION_TYPE, "vnd.openxmlformats-officedocument.wordprocessingml.document");
  public static final MediaType OOXML_PRESENTATION =
      createConstant(
