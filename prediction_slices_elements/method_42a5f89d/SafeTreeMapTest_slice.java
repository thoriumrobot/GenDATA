// Source-based slice around line 106
// Method: <com.google.common.collect.testing.SafeTreeMapTest: void testViewSerialization()>

                MapFeature.GENERAL_PURPOSE,
                CollectionFeature.SUPPORTS_ITERATOR_REMOVE,
                CollectionFeature.SERIALIZABLE)
            .named("SafeTreeMap with null-friendly comparator")
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
}
