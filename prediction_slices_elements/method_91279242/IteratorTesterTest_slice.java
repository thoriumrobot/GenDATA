// Source-based slice around line 180
// Method: <com.google.common.collect.testing.IteratorTesterTest: void testVerifyGetsCalled()>

    }

    @Override
    protected void verify(List<Integer> elements) {
      numCallsToVerify++;
      super.verify(elements);
    }
  }

  public void testVerifyGetsCalled() {
    TesterThatCountsCalls tester = new TesterThatCountsCalls();

    tester.test();

    assertEquals(
        "Should have verified once per stimulus executed",
        tester.numCallsToVerify,
        tester.numCallsToNewTargetIterator * STEPS);
  }

