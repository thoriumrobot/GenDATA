// Source-based slice around line 218
// Method: <com.google.common.collect.testing.IteratorTesterTest: void testMissingException()>

    } catch (AssertionError e) {
      actual = e;
    }
    assertNotNull("verify() should be able to cause test failure", actual);
    assertTrue(
        "AssertionError should have info about why test failed",
        actual.getCause().getMessage().contains(message));
  }

  public void testMissingException() {
    List<Integer> emptyList = new ArrayList<>();

    IteratorTester<Integer> tester =
        new IteratorTester<Integer>(
            1, MODIFIABLE, emptyList, IteratorTester.KnownOrder.KNOWN_ORDER) {
          @Override
          protected Iterator<Integer> newTargetIterator() {
            return new Iterator<Integer>() {
              @Override
              public void remove() {
