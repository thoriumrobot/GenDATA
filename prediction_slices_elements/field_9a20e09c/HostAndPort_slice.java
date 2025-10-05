// Source-based slice around line 69
// Method: com.google.common.net.HostAndPort.NO_PORT

 * caller's responsibility.
 *
 * @author Paul Marks
 * @since 10.0
 */
@Immutable
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

