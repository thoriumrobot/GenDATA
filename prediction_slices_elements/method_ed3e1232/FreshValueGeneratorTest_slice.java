// Source-based slice around line 398
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testTable()>

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
  }

  public void testRowSortedTable() {
    assertFreshInstance(new TypeToken<RowSortedTable<String, ?, ?>>() {});
