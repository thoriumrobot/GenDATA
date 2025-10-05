// Source-based slice around line 220
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder: List getTesters()>

    if (name == null) {
      throw new IllegalStateException("Call named() before createTestSuite().");
    }
    if (features == null) {
      throw new IllegalStateException("Call withFeatures() before createTestSuite().");
    }
  }

  @SuppressWarnings("rawtypes") // class literals
  protected abstract List<Class<? extends AbstractTester>> getTesters();

  private boolean matches(Test test) {
    Method method;
    try {
      method = extractMethod(test);
    } catch (IllegalArgumentException e) {
      logger.finer(Platform.format("%s: including by default: %s", test, e.getMessage()));
      return true;
    }
    if (suppressedTests.contains(method)) {
