// Source-based slice around line 467
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testAddSampleInstances_noInstance()>

  }

  public void testAddSampleInstances_oneInstance() {
    FreshValueGenerator generator = new FreshValueGenerator();
    generator.addSampleInstances(String.class, ImmutableList.of("a"));
    assertEquals("a", generator.generateFresh(String.class));
    assertEquals("a", generator.generateFresh(String.class));
  }

  public void testAddSampleInstances_noInstance() {
    FreshValueGenerator generator = new FreshValueGenerator();
    generator.addSampleInstances(String.class, ImmutableList.<String>of());
    assertEquals(
        new FreshValueGenerator().generateFresh(String.class),
        generator.generateFresh(String.class));
  }

  public void testFreshCurrency() {
    FreshValueGenerator generator = new FreshValueGenerator();
    // repeat a few times to make sure we don't stumble upon a bad Locale
