// Source-based slice around line 86
// Method: <com.google.common.collect.testing.IteratorTesterTest: void testUnknownOrderUnrecognizedElement()>

    new IteratorTester<Integer>(
        3, MODIFIABLE, newArrayList(1, 2), IteratorTester.KnownOrder.UNKNOWN_ORDER) {
      @Override
      protected Iterator<Integer> newTargetIterator() {
        return newArrayList(2, 1).iterator();
      }
    }.test();
  }

  public void testUnknownOrderUnrecognizedElement() {
    IteratorTester<Integer> tester =
        new IteratorTester<Integer>(
            3, MODIFIABLE, newArrayList(1, 2, 50), IteratorTester.KnownOrder.UNKNOWN_ORDER) {
          @Override
          protected Iterator<Integer> newTargetIterator() {
            return newArrayList(2, 1, 3).iterator();
          }
        };
    assertFailure(tester);
  }
