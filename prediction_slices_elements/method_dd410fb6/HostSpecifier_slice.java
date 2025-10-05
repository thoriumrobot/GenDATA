// Source-based slice around line 71
// Method: <com.google.common.net.HostSpecifier: HostSpecifier fromValid(String)>

   * <ul>
   *   <li>A domain name, like {@code google.com}
   *   <li>A IPv4 address string, like {@code 127.0.0.1}
   *   <li>An IPv6 address string with or without brackets, like {@code [2001:db8::1]} or {@code
   *       2001:db8::1}
   * </ul>
   *
   * @throws IllegalArgumentException if the specifier is not valid.
   */
  public static HostSpecifier fromValid(String specifier) {
    // Verify that no port was specified, and strip optional brackets from
    // IPv6 literals.
    HostAndPort parsedHost = HostAndPort.fromString(specifier);
    Preconditions.checkArgument(!parsedHost.hasPort());
    String host = parsedHost.getHost();

    // Try to interpret the specifier as an IP address. Note we build
    // the address rather than using the .is* methods because we want to
    // use InetAddresses.toUriString to convert the result to a string in
    // canonical form.
