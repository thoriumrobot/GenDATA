// Source-based slice around line 176
// Method: com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder.logger

  public B suppressing(Collection<Method> methods) {
    suppressedTests.addAll(methods);
    return self();
  }

  public Set<Method> getSuppressedTests() {
    return suppressedTests;
  }

  private static final Logger logger =
      Logger.getLogger(FeatureSpecificTestSuiteBuilder.class.getName());

  /** Creates a runnable JUnit test suite based on the criteria already given. */
  public TestSuite createTestSuite() {
    checkCanCreate();

    logger.fine(" Testing: " + name);
    logger.fine("Features: " + formatFeatureSet(features));

    addImpliedFeatures(features);
