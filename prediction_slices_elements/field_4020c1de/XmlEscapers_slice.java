// Source-based slice around line 44
// Method: com.google.common.xml.XmlEscapers.MIN_ASCII_CONTROL_CHAR

 *
 * @author Alex Matevossian
 * @author David Beaumont
 * @since 15.0
 */
@GwtCompatible
public class XmlEscapers {
  private XmlEscapers() {}

  private static final char MIN_ASCII_CONTROL_CHAR = 0x00;
  private static final char MAX_ASCII_CONTROL_CHAR = 0x1F;

  // For each xxxEscaper() method, please add links to external reference pages
  // that are considered authoritative for the behavior of that escaper.

  /**
   * Returns an {@link Escaper} instance that escapes special characters in a string so it can
   * safely be included in an XML document as element content. See section <a
   * href="http://www.w3.org/TR/2008/REC-xml-20081126/#syntax">2.4</a> of the XML specification.
   *
