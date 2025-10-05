// Source-based slice around line 207
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder: void checkCanCreate()>

          makeSuiteForTesterClass((Class<? extends AbstractTester<?>>) testerClass);
      if (testerSuite.countTestCases() > 0) {
        suite.addTest(testerSuite);
      }
    }
    return suite;
  }

  /** Throw {@link IllegalStateException} if {@link #createTestSuite()} can't be called yet. */
  protected void checkCanCreate() {
    if (subjectGenerator == null) {
      throw new IllegalStateException("Call using() before createTestSuite().");
    }
    if (name == null) {
      throw new IllegalStateException("Call named() before createTestSuite().");
    }
    if (features == null) {
      throw new IllegalStateException("Call withFeatures() before createTestSuite().");
    }
  }
