// Source-based slice around line 114
// Method: <com.google.common.collect.testing.SafeTreeSetTest: void testSeveral_serialization()>

  @GwtIncompatible // SerializableTester
  public void testSingle_serialization() {
    SortedSet<String> set = new SafeTreeSet<>();
    set.add("e");
    SortedSet<String> copy = SerializableTester.reserializeAndAssert(set);
    assertEquals(set.comparator(), copy.comparator());
  }

  @GwtIncompatible // SerializableTester
  public void testSeveral_serialization() {
    SortedSet<String> set = new SafeTreeSet<>();
    set.add("a");
    set.add("b");
    set.add("c");
    SortedSet<String> copy = SerializableTester.reserializeAndAssert(set);
    assertEquals(set.comparator(), copy.comparator());
  }
}
