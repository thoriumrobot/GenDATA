// Source-based slice around line 278
// Method: <com.google.common.net.HostAndPort: boolean equals(Object)>

   * @throws IllegalArgumentException if bracketless IPv6 is detected.
   */
  @CanIgnoreReturnValue
  public HostAndPort requireBracketsForIPv6() {
    checkArgument(!hasBracketlessColons, "Possible bracketless IPv6 literal: %s", host);
    return this;
  }

  @Override
  public boolean equals(@Nullable Object other) {
    if (this == other) {
      return true;
    }
    if (other instanceof HostAndPort) {
      HostAndPort that = (HostAndPort) other;
      return Objects.equals(this.host, that.host) && this.port == that.port;
    }
    return false;
  }

