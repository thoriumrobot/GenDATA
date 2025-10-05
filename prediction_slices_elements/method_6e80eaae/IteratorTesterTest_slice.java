// Source-based slice around line 140
// Method: <com.google.common.collect.testing.IteratorTesterTest: void testCanCatchJdkBug6529795InTargetIterator()>

    @Override
    public void remove() {
      if (nextThrewException) {
        throw new IllegalStateException();
      }
      iterator.remove();
    }
  }

  public void testCanCatchJdkBug6529795InTargetIterator() {
    try {
      /* Choose 4 steps to get sequence [next, next, next, remove] */
      new IteratorTester<Integer>(
          4, MODIFIABLE, newArrayList(1, 2), IteratorTester.KnownOrder.KNOWN_ORDER) {
        @Override
        protected Iterator<Integer> newTargetIterator() {
          Iterator<Integer> iterator = Lists.newArrayList(1, 2).iterator();
          return new IteratorWithJdkBug6529795<>(iterator);
        }
      }.test();
