// Source-based slice around line 117
// Method: <com.google.common.testing.TearDownStackTest: TearDownStack buildTearDownStack()>

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
          public void tearDown() throws Exception {
            synchronized (result.lock) {
              assertEquals(
                  "The test should have cleared the stack (say, by virtue of running runTearDown)",
                  0,
