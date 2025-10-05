// Source-based slice around line 296
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder: TestSuite filterSuite(TestSuite)>

        @SuppressWarnings("unchecked")
        AbstractTester<? super G> tester = (AbstractTester<? super G>) test;
        tester.init(subjectGenerator, name, setUp, tearDown);
      }
    }

    return suite;
  }

  private TestSuite filterSuite(TestSuite suite) {
    TestSuite filtered = new TestSuite(suite.getName());
    Enumeration<?> tests = suite.tests();
    while (tests.hasMoreElements()) {
      Test test = (Test) tests.nextElement();
      if (matches(test)) {
        filtered.addTest(test);
      }
    }
    return filtered;
  }
