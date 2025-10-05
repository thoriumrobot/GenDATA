// Source-based slice around line 311
// Method: <com.google.common.net.HostAndPort: boolean isValidPort(int)>

      builder.append(host);
    }
    if (hasPort()) {
      builder.append(':').append(port);
    }
    return builder.toString();
  }

  /** Return true for valid port numbers. */
  private static boolean isValidPort(int port) {
    return port >= 0 && port <= 65535;
  }

  @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
}
