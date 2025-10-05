// Source-based slice around line 394
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testHashBiMap()>

  public void testSetMultimap() {
    assertFreshInstance(new TypeToken<SetMultimap<String, ?>>() {});
  }

  public void testBiMap() {
    assertFreshInstance(new TypeToken<BiMap<String, ?>>() {});
    assertNotInstantiable(new TypeToken<BiMap<EmptyEnum, String>>() {});
  }

  public void testHashBiMap() {
    assertFreshInstance(new TypeToken<HashBiMap<String, ?>>() {});
  }

  public void testTable() {
    assertFreshInstance(new TypeToken<Table<String, ?, ?>>() {});
    assertNotInstantiable(new TypeToken<Table<EmptyEnum, String, Integer>>() {});
  }

  public void testHashBasedTable() {
    assertFreshInstance(new TypeToken<HashBasedTable<String, ?, ?>>() {});
