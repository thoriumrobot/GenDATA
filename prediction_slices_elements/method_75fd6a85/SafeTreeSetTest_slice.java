// Source-based slice around line 99
// Method: <com.google.common.collect.testing.SafeTreeSetTest: void testEmpty_serialization()>

    Map<String, Integer> map = ImmutableSortedMap.of("one", 1, "two", 2, "three", 3);
    SerializableTester.reserializeAndAssert(map.entrySet());
    SerializableTester.reserializeAndAssert(map.keySet());
    assertEquals(
        new ArrayList<>(map.values()),
        new ArrayList<>(SerializableTester.reserialize(map.values())));
  }

  @GwtIncompatible // SerializableTester
  public void testEmpty_serialization() {
    SortedSet<String> set = new SafeTreeSet<>();
    SortedSet<String> copy = SerializableTester.reserializeAndAssert(set);
    assertEquals(set.comparator(), copy.comparator());
  }

  @GwtIncompatible // SerializableTester
  public void testSingle_serialization() {
    SortedSet<String> set = new SafeTreeSet<>();
    set.add("e");
    SortedSet<String> copy = SerializableTester.reserializeAndAssert(set);
