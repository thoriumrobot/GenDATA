// Source-based slice around line 116
// Method: com.google.common.net.InternetDomainName.name

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
   * <p>Since this field isn't {@code volatile}, if an instance of this class is shared across
   * threads before it is initialized, then each thread is likely to compute their own copy of the
   * value.
