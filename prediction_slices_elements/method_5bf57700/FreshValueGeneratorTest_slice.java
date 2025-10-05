// Source-based slice around line 403
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testHashBasedTable()>

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
  }

  public void testTreeBasedTable() {
    assertFreshInstance(new TypeToken<TreeBasedTable<String, ?, ?>>() {});
  }
