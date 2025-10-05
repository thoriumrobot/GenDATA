// Source-based slice around line 111
// Method: <com.google.common.net.HostAndPort: int getPort()>

  }

  /**
   * Get the current port number, failing if no port is defined.
   *
   * @return a validated port number, in the range [0..65535]
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
