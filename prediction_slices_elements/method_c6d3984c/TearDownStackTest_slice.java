// Source-based slice around line 75
// Method: <com.google.common.testing.TearDownStackTest: void testThrowingTearDown()>

    assertEquals(false, tearDownOne.ran);
    assertEquals(false, tearDownTwo.ran);

    stack.runTearDown();

    assertEquals("tearDownOne should have run", true, tearDownOne.ran);
    assertEquals("tearDownTwo should have run", true, tearDownTwo.ran);
  }

  public void testThrowingTearDown() throws Exception {
    TearDownStack stack = buildTearDownStack();

    ThrowingTearDown tearDownOne = new ThrowingTearDown("one");
    stack.addTearDown(tearDownOne);

    ThrowingTearDown tearDownTwo = new ThrowingTearDown("two");
    stack.addTearDown(tearDownTwo);

    assertEquals(false, tearDownOne.ran);
    assertEquals(false, tearDownTwo.ran);
