// Source-based slice around line 260
// Method: <com.google.common.collect.testing.IteratorTesterTest: void testSimilarException()>

            1, MODIFIABLE, newArrayList(1), IteratorTester.KnownOrder.KNOWN_ORDER) {
          @Override
          protected Iterator<Integer> newTargetIterator() {
            return new ThrowingIterator<>(new IllegalStateException());
          }
        };
    assertFailure(tester);
  }

  public void testSimilarException() {
    List<Integer> expectedElements = ImmutableList.of();
    IteratorTester<Integer> tester =
        new IteratorTester<Integer>(
            1, MODIFIABLE, expectedElements, IteratorTester.KnownOrder.KNOWN_ORDER) {
          @Override
          protected Iterator<Integer> newTargetIterator() {
            return new Iterator<Integer>() {
              @Override
              public void remove() {
                throw new IllegalStateException() {
