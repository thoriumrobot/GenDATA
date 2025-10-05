// Source-based slice around line 64
// Method: com.google.common.base.Ascii.SOH

  public static final byte NUL = 0;

  /**
   * Start of Heading: A communication control character used at the beginning of a sequence of
   * characters which constitute a machine-sensible address or routing information. Such a sequence
   * is referred to as the "heading." An STX character has the effect of terminating a heading.
   *
   * @since 8.0
   */
  public static final byte SOH = 1;

  /**
   * Start of Text: A communication control character which precedes a sequence of characters that
   * is to be treated as an entity and entirely transmitted through to the ultimate destination.
   * Such a sequence is referred to as "text." STX may be used to terminate a sequence of characters
   * started by SOH.
   *
   * @since 8.0
   */
  public static final byte STX = 2;
