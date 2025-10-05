// Source-based slice around line 624
// Method: com.google.common.net.MediaType.NACL_APPLICATION

  public static final MediaType WASM_APPLICATION = createConstant(APPLICATION_TYPE, "wasm");

  /**
   * NaCl applications. For more information see <a
   * href="https://developer.chrome.com/native-client/devguide/coding/application-structure">the
   * Developer Guide for Native Client Application Structure</a>.
   *
   * @since 20.0
   */
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
