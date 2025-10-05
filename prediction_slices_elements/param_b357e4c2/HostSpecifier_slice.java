// Source-based slice around line 114
// Method: <com.google.common.net.HostSpecifier: HostSpecifier from(String)>


  /**
   * Attempts to return a {@code HostSpecifier} for the given string, throwing an exception if
   * parsing fails. Always use this method in preference to {@link #fromValid(String)} for a
   * specifier that is not already known to be valid.
   *
   * @throws ParseException if the specifier is not valid.
   */
  @CanIgnoreReturnValue // TODO(b/219820829): consider removing
  public static HostSpecifier from(String specifier) throws ParseException {
    try {
      return fromValid(specifier);
    } catch (IllegalArgumentException e) {
      // Since the IAE can originate at several different points inside
      // fromValid(), we implement this method in terms of that one rather
      // than the reverse.

      ParseException parseException = new ParseException("Invalid host specifier: " + specifier, 0);
      parseException.initCause(e);
      throw parseException;
