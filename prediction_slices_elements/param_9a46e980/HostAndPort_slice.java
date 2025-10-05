// Source-based slice around line 133
// Method: <com.google.common.net.HostAndPort: HostAndPort fromParts(String,int)>

   * <p>Note: Non-bracketed IPv6 literals are allowed. Use {@link #requireBracketsForIPv6()} to
   * prohibit these.
   *
   * @param host the host string to parse. Must not contain a port number.
   * @param port a port number from [0..65535]
   * @return if parsing was successful, a populated HostAndPort object.
   * @throws IllegalArgumentException if {@code host} contains a port number, or {@code port} is out
   *     of range.
   */
  public static HostAndPort fromParts(String host, int port) {
    checkArgument(isValidPort(port), "Port out of range: %s", port);
    HostAndPort parsedHost = fromString(host);
    checkArgument(!parsedHost.hasPort(), "Host has a port: %s", host);
    return new HostAndPort(parsedHost.host, port, parsedHost.hasBracketlessColons);
  }

  /**
   * Build a HostAndPort instance from a host only.
   *
   * <p>Note: Non-bracketed IPv6 literals are allowed. Use {@link #requireBracketsForIPv6()} to
