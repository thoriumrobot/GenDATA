// Source-based slice around line 100
// Method: <com.google.common.net.HostAndPort: boolean hasPort()>

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
   *
   * @return a validated port number, in the range [0..65535]
   * @throws IllegalStateException if no port is defined. You can use {@link #withDefaultPort(int)}
   *     to prevent this from occurring.
   */
