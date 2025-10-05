// Source-based slice around line 168
// Method: <com.google.common.net.HostAndPort: HostAndPort fromString(String)>

   *
   * <p>Note that the host-only formats will leave the port field undefined. You can use {@link
   * #withDefaultPort(int)} to patch in a default value.
   *
   * @param hostPortString the input string to parse.
   * @return if parsing was successful, a populated HostAndPort object.
   * @throws IllegalArgumentException if nothing meaningful could be parsed.
   */
  @CanIgnoreReturnValue // TODO(b/219820829): consider removing
  public static HostAndPort fromString(String hostPortString) {
    checkNotNull(hostPortString);
    String host;
    String portString = null;
    boolean hasBracketlessColons = false;

    if (hostPortString.startsWith("[")) {
      String[] hostAndPort = getHostAndPortFromBracketedHost(hostPortString);
      host = hostAndPort[0];
      portString = hostAndPort[1];
    } else {
