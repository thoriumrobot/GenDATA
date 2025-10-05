// Source-based slice around line 571
// Method: <com.google.common.net.InternetDomainName: boolean hasParent()>

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
  }

  /**
   * Returns an {@code InternetDomainName} that is the immediate ancestor of this one; that is, the
   * current domain with the leftmost part removed. For example, the parent of {@code
   * www.google.com} is {@code google.com}.
   *
   * @throws IllegalStateException if the domain has no parent, as determined by {@link #hasParent}
   */
