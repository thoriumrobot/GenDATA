// Source-based slice around line 87
// Method: com.google.common.net.InternetDomainName.NO_SUFFIX_FOUND


  private static final CharMatcher DOTS_MATCHER = CharMatcher.anyOf(".\u3002\uFF0E\uFF61");
  private static final Splitter DOT_SPLITTER = Splitter.on('.');
  private static final Joiner DOT_JOINER = Joiner.on('.');

  /**
   * Value of {@link #publicSuffixIndex()} or {@link #registrySuffixIndex()} which indicates that no
   * relevant suffix was found.
   */
  private static final int NO_SUFFIX_FOUND = -1;

  /**
   * Value of {@link #publicSuffixIndexCache} or {@link #registrySuffixIndexCache} which indicates
   * that they were not initialized yet.
   */
  private static final int SUFFIX_NOT_INITIALIZED = -2;

  /**
   * Maximum parts (labels) in a domain name. This value arises from the 255-octet limit described
   * in <a href="http://www.ietf.org/rfc/rfc2181.txt">RFC 2181</a> part 11 with the fact that the
