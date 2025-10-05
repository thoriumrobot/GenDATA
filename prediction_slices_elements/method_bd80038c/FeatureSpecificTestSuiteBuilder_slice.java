// Source-based slice around line 136
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder: B named(String)>

    return unmodifiableSet(features);
  }

  // Name

  private @Nullable String name;

  /** Configures this builder produce a TestSuite with the given name. */
  @CanIgnoreReturnValue
  public B named(String name) {
    if (name.contains("(")) {
      throw new IllegalArgumentException(
          "Eclipse hides all characters after "
              + "'('; please use '[]' or other characters instead of parentheses");
    }
    this.name = name;
    return self();
  }

  public String getName() {
