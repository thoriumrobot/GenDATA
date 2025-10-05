// Source-based slice around line 97
// Method: <com.google.common.xml.XmlEscapers: Escaper xmlAttributeEscaper()>

   * (NCR). However, horizontal tab {@code '\t'}, line feed {@code '\n'} and carriage return {@code
   * '\r'} are escaped to a corresponding NCR {@code "&#x9;"}, {@code "&#xA;"}, and {@code "&#xD;"}
   * respectively. Any other non-ASCII characters appearing in the input will be preserved in the
   * output.
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
