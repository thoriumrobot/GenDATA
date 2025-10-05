// Source-based slice around line 52
// Method: <com.google.common.collect.testing.IteratorTesterTest: void testCanCatchDifferentContents()>

            4, MODIFIABLE, newArrayList(1, 2, 3), IteratorTester.KnownOrder.KNOWN_ORDER) {
          @Override
          protected Iterator<Integer> newTargetIterator() {
            return Lists.newArrayList(1, 2, 3, 4).iterator();
          }
        };
    assertFailure(tester);
  }

  public void testCanCatchDifferentContents() {
    IteratorTester<Integer> tester =
        new IteratorTester<Integer>(
            3, MODIFIABLE, newArrayList(1, 2, 3), IteratorTester.KnownOrder.KNOWN_ORDER) {
          @Override
          protected Iterator<Integer> newTargetIterator() {
            return Lists.newArrayList(1, 3, 2).iterator();
          }
        };
    assertFailure(tester);
  }
