// Source-based slice around line 76
// Method: <com.google.common.collect.testing.IteratorTesterTest: void testUnknownOrder()>

            3, MODIFIABLE, newArrayList(1, 2), IteratorTester.KnownOrder.KNOWN_ORDER) {
          @Override
          protected Iterator<Integer> newTargetIterator() {
            return ImmutableList.of(1, 2).iterator();
          }
        };
    assertFailure(tester);
  }

  public void testUnknownOrder() {
    new IteratorTester<Integer>(
        3, MODIFIABLE, newArrayList(1, 2), IteratorTester.KnownOrder.UNKNOWN_ORDER) {
      @Override
      protected Iterator<Integer> newTargetIterator() {
        return newArrayList(2, 1).iterator();
      }
    }.test();
  }

  public void testUnknownOrderUnrecognizedElement() {
