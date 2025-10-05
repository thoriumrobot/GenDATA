// Source-based slice around line 221
// Method: com.google.common.base.Ascii.XON

  public static final byte DC1 = 17; // aka XON

  /**
   * Transmission On: Although originally defined as DC1, this ASCII control character is now better
   * known as the XON code used for software flow control in serial communications. The main use is
   * restarting the transmission after the communication has been stopped by the XOFF control code.
   *
   * @since 8.0
   */
  public static final byte XON = 17; // aka DC1

  /**
   * Device Control 2. Characters for the control of ancillary devices associated with data
   * processing or telecommunication systems, more especially switching devices "on" or "off." (If a
   * single "stop" control is required to interrupt or turn off ancillary devices, DC4 is the
   * preferred assignment.)
   *
   * @since 8.0
   */
  public static final byte DC2 = 18;
