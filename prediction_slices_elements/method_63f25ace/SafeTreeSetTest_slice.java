// Source-based slice around line 89
// Method: <com.google.common.collect.testing.SafeTreeSetTest: void testViewSerialization()>

                CollectionFeature.KNOWN_ORDER,
                CollectionFeature.GENERAL_PURPOSE,
                CollectionFeature.ALLOWS_NULL_VALUES)
            .named("SafeTreeSet with null-friendly comparator")
            .createTestSuite());
    return suite;
  }

  @GwtIncompatible // SerializableTester
  public void testViewSerialization() {
    Map<String, Integer> map = ImmutableSortedMap.of("one", 1, "two", 2, "three", 3);
    SerializableTester.reserializeAndAssert(map.entrySet());
    SerializableTester.reserializeAndAssert(map.keySet());
    assertEquals(
        new ArrayList<>(map.values()),
        new ArrayList<>(SerializableTester.reserialize(map.values())));
  }

  @GwtIncompatible // SerializableTester
  public void testEmpty_serialization() {
