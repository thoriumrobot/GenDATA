// Source-based slice around line 64
// Method: <com.google.common.collect.testing.IteratorTesterTest: void testCanCatchDifferentRemoveBehaviour()>

            3, MODIFIABLE, newArrayList(1, 2, 3), IteratorTester.KnownOrder.KNOWN_ORDER) {
          @Override
          protected Iterator<Integer> newTargetIterator() {
            return Lists.newArrayList(1, 3, 2).iterator();
          }
        };
    assertFailure(tester);
  }

  public void testCanCatchDifferentRemoveBehaviour() {
    IteratorTester<Integer> tester =
        new IteratorTester<Integer>(
            3, MODIFIABLE, newArrayList(1, 2), IteratorTester.KnownOrder.KNOWN_ORDER) {
          @Override
          protected Iterator<Integer> newTargetIterator() {
            return ImmutableList.of(1, 2).iterator();
          }
        };
    assertFailure(tester);
  }
