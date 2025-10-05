// Source-based slice around line 152
// Method: com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder.suppressedTests

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
   * <p><em>Note:</em> in principle this should never need to be used, but it might be useful if the
   * semantics of an implementation disagree in unforeseen ways with the semantics expected by a
   * test, or to keep dependent builds clean in spite of an erroneous test.
   */
  @CanIgnoreReturnValue
  public B suppressing(Method... methods) {
