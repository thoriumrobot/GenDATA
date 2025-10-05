// Source-based slice around line 112
// Method: <com.google.common.testing.TearDownStackTest: void tearDown()>

    try {
      setUp();
      runTest();
    } finally {
      tearDown();
    }
  }

  @Override
  protected void tearDown() {
    tearDownStack.runTearDown();
  }

  /** Builds a {@link TearDownStack} that makes sure it's clear by the end of this test. */
  private TearDownStack buildTearDownStack() {
    TearDownStack result = new TearDownStack();
    tearDownStack.addTearDown(
        new TearDown() {

          @Override
