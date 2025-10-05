// Source-based slice around line 267
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder: Method extractMethod(Test)>

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
    } else {
      throw new IllegalArgumentException("unable to extract method from test: not a TestCase.");
    }
  }
