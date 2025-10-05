// Source-based slice around line 315
// Method: com.google.common.net.HostAndPort.serialVersionUID

    }
    return builder.toString();
  }

  /** Return true for valid port numbers. */
  private static boolean isValidPort(int port) {
    return port >= 0 && port <= 65535;
  }

  @GwtIncompatible @J2ktIncompatible private static final long serialVersionUID = 0;
}
