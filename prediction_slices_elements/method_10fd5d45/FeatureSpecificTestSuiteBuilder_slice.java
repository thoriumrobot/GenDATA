// Source-based slice around line 172
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder: Set getSuppressedTests()>

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

  /** Creates a runnable JUnit test suite based on the criteria already given. */
  public TestSuite createTestSuite() {
    checkCanCreate();

