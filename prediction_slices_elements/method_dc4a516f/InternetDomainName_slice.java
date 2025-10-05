// Source-based slice around line 582
// Method: <com.google.common.net.InternetDomainName: InternetDomainName parent()>

  }

  /**
   * Returns an {@code InternetDomainName} that is the immediate ancestor of this one; that is, the
   * current domain with the leftmost part removed. For example, the parent of {@code
   * www.google.com} is {@code google.com}.
   *
   * @throws IllegalStateException if the domain has no parent, as determined by {@link #hasParent}
   */
  public InternetDomainName parent() {
    checkState(hasParent(), "Domain '%s' has no parent", name);
    return ancestor(1);
  }

  /**
   * Returns the ancestor of the current domain at the given number of levels "higher" (rightward)
   * in the subdomain list. The number of levels must be non-negative, and less than {@code N-1},
   * where {@code N} is the number of parts in the domain.
   *
   * <p>TODO: Reasonable candidate for addition to public API.
