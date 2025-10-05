// Source-based slice around line 279
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder: TestSuite makeSuiteForTesterClass(Class)>

      return getMethod(tester.getClass(), tester.getTestMethodName());
    } else if (test instanceof TestCase) {
      TestCase testCase = (TestCase) test;
      return getMethod(testCase.getClass(), testCase.getName());
    } else {
      throw new IllegalArgumentException("unable to extract method from test: not a TestCase.");
    }
  }

  protected TestSuite makeSuiteForTesterClass(Class<? extends AbstractTester<?>> testerClass) {
    TestSuite candidateTests = new TestSuite(testerClass);
    TestSuite suite = filterSuite(candidateTests);

    Enumeration<?> allTests = suite.tests();
    while (allTests.hasMoreElements()) {
      Object test = allTests.nextElement();
      if (test instanceof AbstractTester) {
        @SuppressWarnings("unchecked")
        AbstractTester<? super G> tester = (AbstractTester<? super G>) test;
        tester.init(subjectGenerator, name, setUp, tearDown);
