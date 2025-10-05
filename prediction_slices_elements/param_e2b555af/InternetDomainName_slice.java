// Source-based slice around line 594
// Method: <com.google.common.net.InternetDomainName: InternetDomainName ancestor(int)>

  }

  /**
   * Returns the ancestor of the current domain at the given number of levels "higher" (rightward)
   * in the subdomain list. The number of levels must be non-negative, and less than {@code N-1},
   * where {@code N} is the number of parts in the domain.
   *
   * <p>TODO: Reasonable candidate for addition to public API.
   */
  private InternetDomainName ancestor(int levels) {
    ImmutableList<String> ancestorParts = parts.subList(levels, parts.size());

    // levels equals the number of dots that are getting clipped away, then add the length of each
    // clipped part to get the length of the leading substring that is being removed.
    int substringFrom = levels;
    for (int i = 0; i < levels; i++) {
      substringFrom += parts.get(i).length();
    }
    String ancestorName = name.substring(substringFrom);

