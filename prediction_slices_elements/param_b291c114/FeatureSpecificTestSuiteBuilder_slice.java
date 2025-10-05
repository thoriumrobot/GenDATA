// Source-based slice around line 263
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder: boolean intersect(Set,Set)>

        logger.finer(
            Platform.format(
                "%s: skipping because these features are present: %s", method, unwantedFeatures));
      }
      return false;
    }
    return true;
  }

  private static boolean intersect(Set<?> a, Set<?> b) {
    return !disjoint(a, b);
  }

  private static Method extractMethod(Test test) {
    if (test instanceof AbstractTester) {
      AbstractTester<?> tester = (AbstractTester<?>) test;
      return getMethod(tester.getClass(), tester.getTestMethodName());
    } else if (test instanceof TestCase) {
      TestCase testCase = (TestCase) test;
      return getMethod(testCase.getClass(), testCase.getName());
