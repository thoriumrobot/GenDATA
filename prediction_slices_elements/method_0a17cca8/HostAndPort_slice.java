// Source-based slice around line 290
// Method: <com.google.common.net.HostAndPort: int hashCode()>

    }
    if (other instanceof HostAndPort) {
      HostAndPort that = (HostAndPort) other;
      return Objects.equals(this.host, that.host) && this.port == that.port;
    }
    return false;
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
