// Source-based slice around line 210
// Method: <com.google.common.net.HostAndPort: String[] getHostAndPortFromBracketedHost(String)>

  }

  /**
   * Parses a bracketed host-port string, throwing IllegalArgumentException if parsing fails.
   *
   * @param hostPortString the full bracketed host-port specification. Port might not be specified.
   * @return an array with 2 strings: host and port, in that order.
   * @throws IllegalArgumentException if parsing the bracketed host-port string fails.
   */
  private static String[] getHostAndPortFromBracketedHost(String hostPortString) {
    checkArgument(
        hostPortString.charAt(0) == '[',
        "Bracketed host-port string must start with a bracket: %s",
        hostPortString);
    int colonIndex = hostPortString.indexOf(':');
    int closeBracketIndex = hostPortString.lastIndexOf(']');
    checkArgument(
        colonIndex > -1 && closeBracketIndex > colonIndex,
        "Invalid bracketed host/port: %s",
        hostPortString);
