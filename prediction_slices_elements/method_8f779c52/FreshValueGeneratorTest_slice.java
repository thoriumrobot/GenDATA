// Source-based slice around line 483
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testNulls()>


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
        .testNulls(FreshValueGenerator.class);
  }

  private static void assertFreshInstances(Class<?>... types) {
    for (Class<?> type : types) {
      assertFreshInstance(type, 2);
    }
  }
