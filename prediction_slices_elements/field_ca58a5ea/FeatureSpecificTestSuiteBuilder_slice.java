// Source-based slice around line 132
// Method: com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder.name

    return self();
  }

  public Set<Feature<?>> getFeatures() {
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
