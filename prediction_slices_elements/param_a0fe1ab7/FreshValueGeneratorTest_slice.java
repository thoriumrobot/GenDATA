// Source-based slice around line 489
// Method: <com.google.common.testing.FreshValueGeneratorTest: void assertFreshInstances(Class)>

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

  private static void assertFreshInstance(TypeToken<?> type) {
    assertFreshInstance(type, 3);
  }

  private static void assertFreshInstance(Class<?> type, int instances) {
