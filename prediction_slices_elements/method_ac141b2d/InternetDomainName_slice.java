// Source-based slice around line 213
// Method: <com.google.common.net.InternetDomainName: int findSuffixOfType(Optional)>

  /**
   * Returns the index of the leftmost part of the suffix, or -1 if not found. Note that the value
   * defined as a suffix may not produce {@code true} results from {@link #isPublicSuffix()} or
   * {@link #isRegistrySuffix()} if the domain ends with an excluded domain pattern such as {@code
   * "nhs.uk"}.
   *
   * <p>If a {@code desiredType} is specified, this method only finds suffixes of the given type.
   * Otherwise, it finds the first suffix of any type.
   */
  private int findSuffixOfType(Optional<PublicSuffixType> desiredType) {
    int partsSize = parts.size();

    for (int i = 0; i < partsSize; i++) {
      String ancestorName = DOT_JOINER.join(parts.subList(i, partsSize));

      if (i > 0
          && matchesType(
              desiredType, Optional.fromNullable(PublicSuffixPatterns.UNDER.get(ancestorName)))) {
        return i - 1;
      }
