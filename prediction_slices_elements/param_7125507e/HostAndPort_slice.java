// Source-based slice around line 151
// Method: <com.google.common.net.HostAndPort: HostAndPort fromHost(String)>

   *
   * <p>Note: Non-bracketed IPv6 literals are allowed. Use {@link #requireBracketsForIPv6()} to
   * prohibit these.
   *
   * @param host the host-only string to parse. Must not contain a port number.
   * @return if parsing was successful, a populated HostAndPort object.
   * @throws IllegalArgumentException if {@code host} contains a port number.
   * @since 17.0
   */
  public static HostAndPort fromHost(String host) {
    HostAndPort parsedHost = fromString(host);
    checkArgument(!parsedHost.hasPort(), "Host has a port: %s", host);
    return parsedHost;
  }

  /**
   * Split a freeform string into a host and port, without strict validation.
   *
   * <p>Note that the host-only formats will leave the port field undefined. You can use {@link
   * #withDefaultPort(int)} to patch in a default value.
