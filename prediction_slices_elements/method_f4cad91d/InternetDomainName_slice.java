// Source-based slice around line 644
// Method: <com.google.common.net.InternetDomainName: boolean isValid(String)>

   * try {
   *   domainName = InternetDomainName.from(name);
   * } catch (IllegalArgumentException e) {
   *   domainName = DEFAULT_DOMAIN;
   * }
   * }
   *
   * @since 8.0 (previously named {@code isValidLenient})
   */
  public static boolean isValid(String name) {
    try {
      InternetDomainName unused = from(name);
      return true;
    } catch (IllegalArgumentException e) {
      return false;
    }
  }

  /**
   * If a {@code desiredType} is specified, returns true only if the {@code actualType} is
