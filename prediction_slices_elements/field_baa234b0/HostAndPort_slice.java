// Source-based slice around line 75
// Method: com.google.common.net.HostAndPort.port

@GwtCompatible
public final class HostAndPort implements Serializable {
  /** Magic value indicating the absence of a port number. */
  private static final int NO_PORT = -1;

  /** Hostname, IPv4/IPv6 literal, or unvalidated nonsense. */
  private final String host;

  /** Validated port number in the range [0..65535], or NO_PORT */
  private final int port;

  /** True if the parsed host has colons, but no surrounding brackets. */
  private final boolean hasBracketlessColons;

  private HostAndPort(String host, int port, boolean hasBracketlessColons) {
    this.host = host;
    this.port = port;
    this.hasBracketlessColons = hasBracketlessColons;
  }

