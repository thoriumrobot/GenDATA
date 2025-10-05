// Source-based slice around line 93
// Method: com.google.common.net.InternetDomainName.SUFFIX_NOT_INITIALIZED

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
   * encoding of each part occupies at least two bytes (dot plus label externally, length byte plus
   * label internally). Thus, if all labels have the minimum size of one byte, 127 of them will fit.
   */
  private static final int MAX_PARTS = 127;

  /**
