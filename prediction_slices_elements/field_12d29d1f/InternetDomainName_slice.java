// Source-based slice around line 113
// Method: com.google.common.net.InternetDomainName.MAX_DOMAIN_PART_LENGTH

   * Maximum length of a full domain name, including separators, and leaving room for the root
   * label. See <a href="http://www.ietf.org/rfc/rfc2181.txt">RFC 2181</a> part 11.
   */
  private static final int MAX_LENGTH = 253;

  /**
   * Maximum size of a single part of a domain name. See <a
   * href="http://www.ietf.org/rfc/rfc2181.txt">RFC 2181</a> part 11.
   */
  private static final int MAX_DOMAIN_PART_LENGTH = 63;

  /** The full domain name, converted to lower case. */
  private final String name;

  /** The parts of the domain name, converted to lower case. */
  private final ImmutableList<String> parts;

  /**
   * Cached value of #publicSuffixIndex(). Do not use directly.
   *
