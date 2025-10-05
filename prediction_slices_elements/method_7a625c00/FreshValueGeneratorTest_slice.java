// Source-based slice around line 431
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testGoogleOptional()>

    assertEqualInstance(EmptyEnum.class, null);
    assertEqualInstance(OneConstantEnum.class, OneConstantEnum.CONSTANT1);
    assertFreshInstance(TwoConstantEnum.class, 2);
    assertFreshInstance(new TypeToken<com.google.common.base.Optional<OneConstantEnum>>() {}, 2);
    assertFreshInstance(new TypeToken<List<OneConstantEnum>>() {}, 1);
    assertFreshInstance(new TypeToken<List<TwoConstantEnum>>() {}, 2);
  }

  @AndroidIncompatible // problem with equality of Type objects?
  public void testGoogleOptional() {
    FreshValueGenerator generator = new FreshValueGenerator();
    assertEquals(
        com.google.common.base.Optional.absent(),
        generator.generateFresh(new TypeToken<com.google.common.base.Optional<String>>() {}));
    assertEquals(
        com.google.common.base.Optional.of("2"),
        generator.generateFresh(new TypeToken<com.google.common.base.Optional<String>>() {}));
    // Test that the first generated instance for different cgcb.Optional<T> is always absent().
    // Having generated cgcb.Optional<String> instances doesn't prevent absent() from being
    // generated for other cgcb.Optional types.
