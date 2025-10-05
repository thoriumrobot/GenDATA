// Source-based slice around line 88
// Method: <com.google.common.testing.TestLogHandlerTest: void tearDown()>

    try {
      setUp();
      runTest();
    } finally {
      tearDown();
    }
  }

  @Override
  protected void tearDown() {
    stack.runTearDown();
  }

  static final Exception EXCEPTION = new Exception();

  static final class ExampleClassUnderTest {
    static final Logger logger = Logger.getLogger(ExampleClassUnderTest.class.getName());

    static void foo() {
      logger.log(Level.INFO, "message", EXCEPTION);
