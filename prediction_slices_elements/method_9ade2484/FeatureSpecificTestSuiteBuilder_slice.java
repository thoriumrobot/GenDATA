// Source-based slice around line 146
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder: String getName()>

    if (name.contains("(")) {
      throw new IllegalArgumentException(
          "Eclipse hides all characters after "
              + "'('; please use '[]' or other characters instead of parentheses");
    }
    this.name = name;
    return self();
  }

  public String getName() {
    return name;
  }

  // Test suppression

  private final Set<Method> suppressedTests = new HashSet<>();

  /**
   * Prevents the given methods from being run as part of the test suite.
   *
