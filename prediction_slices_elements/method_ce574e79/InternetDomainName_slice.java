// Source-based slice around line 528
// Method: <com.google.common.net.InternetDomainName: boolean isUnderRegistrySuffix()>


  /**
   * Indicates whether this domain name ends in a {@linkplain #isRegistrySuffix() registry suffix},
   * while not being a registry suffix itself. For example, returns {@code true} for {@code
   * www.google.com}, {@code foo.co.uk} and {@code blogspot.com}, but not for {@code com}, {@code
   * co.uk}, or {@code google.invalid}.
   *
   * @since 23.3
   */
  public boolean isUnderRegistrySuffix() {
    return registrySuffixIndex() > 0;
  }

  /**
   * Indicates whether this domain name is composed of exactly one subdomain component followed by a
   * {@linkplain #isRegistrySuffix() registry suffix}. For example, returns {@code true} for {@code
   * google.com}, {@code foo.co.uk}, and {@code blogspot.com}, but not for {@code www.google.com},
   * {@code co.uk}, or {@code myblog.blogspot.com}.
   *
   * <p><b>Warning:</b> This method should not be used to determine the probable highest level
