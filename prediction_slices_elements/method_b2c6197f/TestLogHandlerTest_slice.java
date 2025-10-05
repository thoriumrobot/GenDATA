// Source-based slice around line 66
// Method: <com.google.common.testing.TestLogHandlerTest: void testConcurrentModification()>

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
    // stored log records causes a ConcurrentModificationException
    assertTrue(handler.getStoredLogRecords().isEmpty());
    ExampleClassUnderTest.foo();
    ExampleClassUnderTest.foo();
    for (LogRecord unused : handler.getStoredLogRecords()) {
      ExampleClassUnderTest.foo();
    }
  }

