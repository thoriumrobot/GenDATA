// Source-based slice around line 117
// Method: <com.google.common.net.HostAndPort: int getPortOrDefault(int)>

   * @throws IllegalStateException if no port is defined. You can use {@link #withDefaultPort(int)}
   *     to prevent this from occurring.
   */
  public int getPort() {
    checkState(hasPort());
    return port;
  }

  /** Returns the current port number, with a default if no port is defined. */
  public int getPortOrDefault(int defaultPort) {
    return hasPort() ? port : defaultPort;
  }

  /**
   * Build a HostAndPort instance from separate host and port values.
   *
   * <p>Note: Non-bracketed IPv6 literals are allowed. Use {@link #requireBracketsForIPv6()} to
   * prohibit these.
   *
   * @param host the host string to parse. Must not contain a port number.
