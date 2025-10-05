// Source-based slice around line 156
// Method: <com.google.common.net.HostSpecifier: int hashCode()>

    if (other instanceof HostSpecifier) {
      HostSpecifier that = (HostSpecifier) other;
      return this.canonicalForm.equals(that.canonicalForm);
    }

    return false;
  }

  @Override
  public int hashCode() {
    return canonicalForm.hashCode();
  }

  /**
   * Returns a string representation of the host specifier suitable for inclusion in a URI. If the
   * host specifier is a domain name, the string will be normalized to all lower case. If the
   * specifier was an IPv6 address without brackets, brackets are added so that the result will be
   * usable in the host part of a URI.
   */
  @Override
