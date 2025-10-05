// Source-based slice around line 562
// Method: <com.google.common.net.InternetDomainName: InternetDomainName topDomainUnderRegistrySuffix()>

   * <p>If {@link #isTopDomainUnderRegistrySuffix()} is true, the current domain name instance is
   * returned.
   *
   * <p><b>Warning:</b> This method should not be used to determine whether a domain is probably the
   * highest level for which cookies may be set. Use {@link #isTopPrivateDomain()} for that purpose.
   *
   * @throws IllegalStateException if this domain does not end with a registry suffix
   * @since 23.3
   */
  public InternetDomainName topDomainUnderRegistrySuffix() {
    if (isTopDomainUnderRegistrySuffix()) {
      return this;
    }
    checkState(isUnderRegistrySuffix(), "Not under a registry suffix: %s", name);
    return ancestor(registrySuffixIndex() - 1);
  }

  /** Indicates whether this domain is composed of two or more parts. */
  public boolean hasParent() {
    return parts.size() > 1;
