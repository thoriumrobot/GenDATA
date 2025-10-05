// Source-based slice around line 460
// Method: <com.google.common.net.InternetDomainName: InternetDomainName topPrivateDomain()>

   * <p>If {@link #isTopPrivateDomain()} is true, the current domain name instance is returned.
   *
   * <p>This method can be used to determine the probable highest level parent domain for which
   * cookies may be set, though even that depends on individual browsers' implementations of cookie
   * controls.
   *
   * @throws IllegalStateException if this domain does not end with a public suffix
   * @since 6.0
   */
  public InternetDomainName topPrivateDomain() {
    if (isTopPrivateDomain()) {
      return this;
    }
    checkState(isUnderPublicSuffix(), "Not under a public suffix: %s", name);
    return ancestor(publicSuffixIndex() - 1);
  }

  /**
   * Indicates whether this domain name represents a <i>registry suffix</i>, as defined by a subset
   * of the Mozilla Foundation's <a href="http://publicsuffix.org/">Public Suffix List</a> (PSL). A
