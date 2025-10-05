// Source-based slice around line 128
// Method: <com.google.common.base.StandardSystemProperty: String key()>

  USER_DIR("user.dir");

  private final String key;

  StandardSystemProperty(String key) {
    this.key = key;
  }

  /** Returns the key used to look up this system property. */
  public String key() {
    return key;
  }

  /**
   * Returns the current value for this system property by delegating to {@link
   * System#getProperty(String)}.
   *
   * <p>The value returned by this method is non-null except in rare circumstances:
   *
   * <ul>
