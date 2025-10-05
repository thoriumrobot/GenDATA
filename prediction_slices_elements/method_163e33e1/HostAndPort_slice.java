// Source-based slice around line 272
// Method: <com.google.common.net.HostAndPort: HostAndPort requireBracketsForIPv6()>

   *
   * <p>Note that this parser identifies IPv6 literals solely based on the presence of a colon. To
   * perform actual validation of IP addresses, see the {@link InetAddresses#forString(String)}
   * method.
   *
   * @return {@code this}, to enable chaining of calls.
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
