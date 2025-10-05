// Source-based slice around line 180
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder: TestSuite createTestSuite()>


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

    logger.fine("Expanded: " + formatFeatureSet(features));

    @SuppressWarnings("rawtypes") // class literals
