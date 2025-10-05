// Source-based slice around line 415
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testObject()>


  public void testRowSortedTable() {
    assertFreshInstance(new TypeToken<RowSortedTable<String, ?, ?>>() {});
  }

  public void testTreeBasedTable() {
    assertFreshInstance(new TypeToken<TreeBasedTable<String, ?, ?>>() {});
  }

  public void testObject() {
    assertEquals(
        new FreshValueGenerator().generateFresh(String.class),
        new FreshValueGenerator().generateFresh(Object.class));
  }

  public void testEnums() {
    assertEqualInstance(EmptyEnum.class, null);
    assertEqualInstance(OneConstantEnum.class, OneConstantEnum.CONSTANT1);
    assertFreshInstance(TwoConstantEnum.class, 2);
    assertFreshInstance(new TypeToken<com.google.common.base.Optional<OneConstantEnum>>() {}, 2);
