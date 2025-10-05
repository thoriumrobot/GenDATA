// Source-based slice around line 249
// Method: <com.google.common.net.HostAndPort: HostAndPort withDefaultPort(int)>

  /**
   * Provide a default port if the parsed string contained only a host.
   *
   * <p>You can chain this after {@link #fromString(String)} to include a port in case the port was
   * omitted from the input string. If a port was already provided, then this method is a no-op.
   *
   * @param defaultPort a port number, from [0..65535]
   * @return a HostAndPort instance, guaranteed to have a defined port.
   */
  public HostAndPort withDefaultPort(int defaultPort) {
    checkArgument(isValidPort(defaultPort));
    if (hasPort()) {
      return this;
    }
    return new HostAndPort(host, defaultPort, hasBracketlessColons);
  }

  /**
   * Generate an error if the host might be a non-bracketed IPv6 literal.
   *
