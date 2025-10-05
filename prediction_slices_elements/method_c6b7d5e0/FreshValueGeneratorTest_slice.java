// Source-based slice around line 365
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testMultimap()>

    assertFreshInstance(new TypeToken<NavigableMap<?, ?>>() {});
  }

  public void testConcurrentMap() {
    assertFreshInstance(new TypeToken<ConcurrentMap<String, ?>>() {});
    assertCanGenerateOnly(
        new TypeToken<ConcurrentMap<EmptyEnum, String>>() {}, Maps.newConcurrentMap());
  }

  public void testMultimap() {
    assertFreshInstance(new TypeToken<Multimap<String, ?>>() {});
  }

  public void testHashMultimap() {
    assertFreshInstance(new TypeToken<HashMultimap<String, ?>>() {});
  }

  public void testLinkedHashMultimap() {
    assertFreshInstance(new TypeToken<LinkedHashMultimap<String, ?>>() {});
  }
