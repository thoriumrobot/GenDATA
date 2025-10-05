// Source-based slice around line 292
// Method: <com.google.common.collect.testing.IteratorTesterTest: void testMismatchedException()>

              public boolean hasNext() {
                return false;
              }
            };
          }
        };
    tester.test();
  }

  public void testMismatchedException() {
    List<Integer> expectedElements = ImmutableList.of();
    IteratorTester<Integer> tester =
        new IteratorTester<Integer>(
            1, MODIFIABLE, expectedElements, IteratorTester.KnownOrder.KNOWN_ORDER) {
          @Override
          protected Iterator<Integer> newTargetIterator() {
            return new Iterator<Integer>() {
              @Override
              public void remove() {
                // Wrong exception type.
