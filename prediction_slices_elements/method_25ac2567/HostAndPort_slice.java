// Source-based slice around line 296
// Method: <com.google.common.net.HostAndPort: String toString()>

  }

  @Override
  public int hashCode() {
    return Objects.hash(host, port);
  }

  /** Rebuild the host:port string, including brackets if necessary. */
  @Override
  public String toString() {
    // "[]:12345" requires 8 extra bytes.
    StringBuilder builder = new StringBuilder(host.length() + 8);
    if (host.indexOf(':') >= 0) {
      builder.append('[').append(host).append(']');
    } else {
      builder.append(host);
    }
    if (hasPort()) {
      builder.append(':').append(port);
    }
