// Source-based slice around line 674
// Method: <com.google.common.net.InternetDomainName: boolean equals(Object)>

    return name;
  }

  /**
   * Equality testing is based on the text supplied by the caller, after normalization as described
   * in the class documentation. For example, a non-ASCII Unicode domain name and the Punycode
   * version of the same domain name would not be considered equal.
   */
  @Override
  public boolean equals(@Nullable Object object) {
    if (object == this) {
      return true;
    }

    if (object instanceof InternetDomainName) {
      InternetDomainName that = (InternetDomainName) object;
      return this.name.equals(that.name);
    }

    return false;
