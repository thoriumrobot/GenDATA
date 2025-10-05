// Source-based slice around line 95
// Method: <com.google.common.net.HostAndPort: String getHost()>

  /**
   * Returns the portion of this {@code HostAndPort} instance that should represent the hostname or
   * IPv4/IPv6 literal.
   *
   * <p>A successful parse does not imply any degree of sanity in this field. For additional
   * validation, see the {@link HostSpecifier} class.
   *
   * @since 20.0 (since 10.0 as {@code getHostText})
   */
  public String getHost() {
    return host;
  }

  /** Return true if this instance has a defined port. */
  public boolean hasPort() {
    return port >= 0;
  }

  /**
   * Get the current port number, failing if no port is defined.
