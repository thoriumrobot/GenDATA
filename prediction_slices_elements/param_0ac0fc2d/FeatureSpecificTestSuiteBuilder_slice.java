// Source-based slice around line 167
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder: B suppressing(Collection)>

   * semantics of an implementation disagree in unforeseen ways with the semantics expected by a
   * test, or to keep dependent builds clean in spite of an erroneous test.
   */
  @CanIgnoreReturnValue
  public B suppressing(Method... methods) {
    return suppressing(asList(methods));
  }

  @CanIgnoreReturnValue
  public B suppressing(Collection<Method> methods) {
    suppressedTests.addAll(methods);
    return self();
  }

  public Set<Method> getSuppressedTests() {
    return suppressedTests;
  }

  private static final Logger logger =
      Logger.getLogger(FeatureSpecificTestSuiteBuilder.class.getName());
