// Source-based slice around line 167
// Method: <com.google.common.net.HostSpecifier: String toString()>

  }

  /**
   * Returns a string representation of the host specifier suitable for inclusion in a URI. If the
   * host specifier is a domain name, the string will be normalized to all lower case. If the
   * specifier was an IPv6 address without brackets, brackets are added so that the result will be
   * usable in the host part of a URI.
   */
  @Override
  public String toString() {
    return canonicalForm;
  }
}
