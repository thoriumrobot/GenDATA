// Source-based slice around line 57
// Method: <com.google.common.testing.TestLogHandlerTest: void test()>

        new TearDown() {
          @Override
          public void tearDown() throws Exception {
            ExampleClassUnderTest.logger.setUseParentHandlers(true);
            ExampleClassUnderTest.logger.removeHandler(handler);
          }
        });
  }

  public void test() throws Exception {
    assertTrue(handler.getStoredLogRecords().isEmpty());
    ExampleClassUnderTest.foo();
    LogRecord record = handler.getStoredLogRecords().get(0);
    assertEquals(Level.INFO, record.getLevel());
    assertEquals("message", record.getMessage());
    assertSame(EXCEPTION, record.getThrown());
  }

  public void testConcurrentModification() throws Exception {
    // Tests for the absence of a bug where logging while iterating over the
