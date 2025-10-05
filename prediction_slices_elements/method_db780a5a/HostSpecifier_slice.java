// Source-based slice around line 142
// Method: <com.google.common.net.HostSpecifier: boolean equals(Object)>

    try {
      HostSpecifier unused = fromValid(specifier);
      return true;
    } catch (IllegalArgumentException e) {
      return false;
    }
  }

  @Override
  public boolean equals(@Nullable Object other) {
    if (this == other) {
      return true;
    }

    if (other instanceof HostSpecifier) {
      HostSpecifier that = (HostSpecifier) other;
      return this.canonicalForm.equals(that.canonicalForm);
    }

    return false;
