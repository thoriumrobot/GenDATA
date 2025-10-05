// Source-based slice around line 452
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testAddSampleInstances_twoInstances()>

        com.google.common.base.Optional.absent(),
        generator.generateFresh(
            new TypeToken<com.google.common.base.Optional<OneConstantEnum>>() {}));
    assertEquals(
        com.google.common.base.Optional.of(OneConstantEnum.CONSTANT1),
        generator.generateFresh(
            new TypeToken<com.google.common.base.Optional<OneConstantEnum>>() {}));
  }

  public void testAddSampleInstances_twoInstances() {
    FreshValueGenerator generator = new FreshValueGenerator();
    generator.addSampleInstances(String.class, ImmutableList.of("a", "b"));
    assertEquals("a", generator.generateFresh(String.class));
    assertEquals("b", generator.generateFresh(String.class));
    assertEquals("a", generator.generateFresh(String.class));
  }

  public void testAddSampleInstances_oneInstance() {
    FreshValueGenerator generator = new FreshValueGenerator();
    generator.addSampleInstances(String.class, ImmutableList.of("a"));
