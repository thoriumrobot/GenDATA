// Source-based slice around line 101
// Method: com.google.common.xml.XmlEscapers.XML_ATTRIBUTE_ESCAPER

   *
   * <p>This escaper does not treat surrogate pairs specially and does not perform Unicode
   * validation on its input.
   */
  @SuppressWarnings("EscapedEntity") // We do mean for the user to see &#x9;" etc.
  public static Escaper xmlAttributeEscaper() {
    return XML_ATTRIBUTE_ESCAPER;
  }
  private static final Escaper XML_CONTENT_ESCAPER;
  private static final Escaper XML_ATTRIBUTE_ESCAPER;

  static {
    Escapers.Builder builder = Escapers.builder();
    // The char values \uFFFE and \uFFFF are explicitly not allowed in XML
    // (Unicode code points above \uFFFF are represented via surrogate pairs
    // which means they are treated as pairs of safe characters).
    builder.setSafeRange(Character.MIN_VALUE, '\uFFFD');
    // Unsafe characters are replaced with the Unicode replacement character.
    builder.setUnsafeReplacement("\uFFFD");

