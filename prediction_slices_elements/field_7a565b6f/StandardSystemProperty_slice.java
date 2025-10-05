// Source-based slice around line 121
// Method: com.google.common.base.StandardSystemProperty.key

  /** User's account name. */
  USER_NAME("user.name"),

  /** User's home directory. */
  USER_HOME("user.home"),

  /** User's current working directory. */
  USER_DIR("user.dir");

  private final String key;

  StandardSystemProperty(String key) {
    this.key = key;
  }

  /** Returns the key used to look up this system property. */
  public String key() {
    return key;
  }

