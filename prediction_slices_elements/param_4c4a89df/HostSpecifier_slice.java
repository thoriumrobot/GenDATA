// Source-based slice around line 132
// Method: <com.google.common.net.HostSpecifier: boolean isValid(String)>

      parseException.initCause(e);
      throw parseException;
    }
  }

  /**
   * Determines whether {@code specifier} represents a valid {@link HostSpecifier} as described in
   * the documentation for {@link #fromValid(String)}.
   */
  public static boolean isValid(String specifier) {
    try {
      HostSpecifier unused = fromValid(specifier);
      return true;
    } catch (IllegalArgumentException e) {
      return false;
    }
  }

  @Override
  public boolean equals(@Nullable Object other) {
