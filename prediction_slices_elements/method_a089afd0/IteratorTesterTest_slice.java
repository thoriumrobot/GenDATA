// Source-based slice around line 191
// Method: <com.google.common.collect.testing.IteratorTesterTest: void testVerifyCanThrowAssertionThatFailsTest()>


    tester.test();

    assertEquals(
        "Should have verified once per stimulus executed",
        tester.numCallsToVerify,
        tester.numCallsToNewTargetIterator * STEPS);
  }

  public void testVerifyCanThrowAssertionThatFailsTest() {
    String message = "Important info about why verify failed";
    IteratorTester<Integer> tester =
        new IteratorTester<Integer>(
            1, MODIFIABLE, newArrayList(1, 2, 3), IteratorTester.KnownOrder.KNOWN_ORDER) {
          @Override
          protected Iterator<Integer> newTargetIterator() {
            return Lists.newArrayList(1, 2, 3).iterator();
          }

          @Override
