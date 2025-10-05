// Source-based slice around line 48
// Method: <com.google.common.testing.TearDownStackTest: void testMultipleTearDownsHappenInOrder()>

    stack.addTearDown(tearDown);

    assertEquals(false, tearDown.ran);

    stack.runTearDown();

    assertEquals("tearDown should have run", true, tearDown.ran);
  }

  public void testMultipleTearDownsHappenInOrder() throws Exception {
    TearDownStack stack = buildTearDownStack();

    SimpleTearDown tearDownOne = new SimpleTearDown();
    stack.addTearDown(tearDownOne);

    Callback callback =
        new Callback() {
          @Override
          public void run() {
            assertEquals(
