// Source-based slice around line 475
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testFreshCurrency()>


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
    assertNotNull(generator.generateFresh(Currency.class));
    assertNotNull(generator.generateFresh(Currency.class));
    assertNotNull(generator.generateFresh(Currency.class));
  }

  public void testNulls() throws Exception {
    new ClassSanityTester()
        .setDefault(Method.class, FreshValueGeneratorTest.class.getDeclaredMethod("testNulls"))
