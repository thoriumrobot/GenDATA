// Source-based slice around line 101
// Method: com.google.common.net.InternetDomainName.MAX_PARTS

   */
  private static final int SUFFIX_NOT_INITIALIZED = -2;

  /**
   * Maximum parts (labels) in a domain name. This value arises from the 255-octet limit described
   * in <a href="http://www.ietf.org/rfc/rfc2181.txt">RFC 2181</a> part 11 with the fact that the
   * encoding of each part occupies at least two bytes (dot plus label externally, length byte plus
   * label internally). Thus, if all labels have the minimum size of one byte, 127 of them will fit.
   */
  private static final int MAX_PARTS = 127;

  /**
   * Maximum length of a full domain name, including separators, and leaving room for the root
   * label. See <a href="http://www.ietf.org/rfc/rfc2181.txt">RFC 2181</a> part 11.
   */
  private static final int MAX_LENGTH = 253;

  /**
   * Maximum size of a single part of a domain name. See <a
   * href="http://www.ietf.org/rfc/rfc2181.txt">RFC 2181</a> part 11.
