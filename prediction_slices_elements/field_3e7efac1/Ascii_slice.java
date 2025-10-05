// Source-based slice around line 202
// Method: com.google.common.base.Ascii.DLE

  public static final byte SI = 15;

  /**
   * Data Link Escape: A communication control character which will change the meaning of a limited
   * number of contiguously following characters. It is used exclusively to provide supplementary
   * controls in data communication networks.
   *
   * @since 8.0
   */
  public static final byte DLE = 16;

  /**
   * Device Control 1. Characters for the control of ancillary devices associated with data
   * processing or telecommunication systems, more especially switching devices "on" or "off." (If a
   * single "stop" control is required to interrupt or turn off ancillary devices, DC4 is the
   * preferred assignment.)
   *
   * @since 8.0
   */
  public static final byte DC1 = 17; // aka XON
