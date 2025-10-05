// Source-based slice around line 270
// Method: <com.google.common.net.InternetDomainName: boolean validateSyntax(List)>

    return new InternetDomainName(checkNotNull(domain));
  }

  /**
   * Validation method used by {@code from} to ensure that the domain name is syntactically valid
   * according to RFC 1035.
   *
   * @return Is the domain name syntactically valid?
   */
  private static boolean validateSyntax(List<String> parts) {
    int lastIndex = parts.size() - 1;

    // Validate the last part specially, as it has different syntax rules.

    if (!validatePart(parts.get(lastIndex), true)) {
      return false;
    }

    for (int i = 0; i < lastIndex; i++) {
      String part = parts.get(i);
