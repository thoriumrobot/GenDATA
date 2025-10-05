// Source-based slice around line 664
// Method: <com.google.common.net.InternetDomainName: String toString()>

   * identical. Otherwise, returns true as long as {@code actualType} is present.
   */
  private static boolean matchesType(
      Optional<PublicSuffixType> desiredType, Optional<PublicSuffixType> actualType) {
    return desiredType.isPresent() ? desiredType.equals(actualType) : actualType.isPresent();
  }

  /** Returns the domain name, normalized to all lower case. */
  @Override
  public String toString() {
    return name;
  }

  /**
   * Equality testing is based on the text supplied by the caller, after normalization as described
   * in the class documentation. For example, a non-ASCII Unicode domain name and the Punycode
   * version of the same domain name would not be considered equal.
   */
  @Override
  public boolean equals(@Nullable Object object) {
