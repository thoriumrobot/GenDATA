// Source-based slice around line 460
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testAddSampleInstances_oneInstance()>


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
    assertEquals("a", generator.generateFresh(String.class));
    assertEquals("a", generator.generateFresh(String.class));
  }

  public void testAddSampleInstances_noInstance() {
    FreshValueGenerator generator = new FreshValueGenerator();
    generator.addSampleInstances(String.class, ImmutableList.<String>of());
    assertEquals(
