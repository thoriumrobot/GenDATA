// Source-based slice around line 421
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testEnums()>

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
    assertFreshInstance(new TypeToken<List<OneConstantEnum>>() {}, 1);
    assertFreshInstance(new TypeToken<List<TwoConstantEnum>>() {}, 2);
  }

  @AndroidIncompatible // problem with equality of Type objects?
  public void testGoogleOptional() {
