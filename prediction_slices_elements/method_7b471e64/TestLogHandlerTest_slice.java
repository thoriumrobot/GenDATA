// Source-based slice around line 78
// Method: <com.google.common.testing.TestLogHandlerTest: void runBare()>

    assertTrue(handler.getStoredLogRecords().isEmpty());
    ExampleClassUnderTest.foo();
    ExampleClassUnderTest.foo();
    for (LogRecord unused : handler.getStoredLogRecords()) {
      ExampleClassUnderTest.foo();
    }
  }

  @Override
  public final void runBare() throws Throwable {
    try {
      setUp();
      runTest();
    } finally {
      tearDown();
    }
  }

  @Override
  protected void tearDown() {
