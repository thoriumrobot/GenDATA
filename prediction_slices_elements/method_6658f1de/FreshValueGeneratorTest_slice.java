// Source-based slice around line 503
// Method: <com.google.common.testing.FreshValueGeneratorTest: void assertFreshInstance(TypeToken,int)>


  private static void assertFreshInstance(TypeToken<?> type) {
    assertFreshInstance(type, 3);
  }

  private static void assertFreshInstance(Class<?> type, int instances) {
    assertFreshInstance(TypeToken.of(type), instances);
  }

  private static void assertFreshInstance(TypeToken<?> type, int instances) {
    FreshValueGenerator generator = new FreshValueGenerator();
    EqualsTester tester = new EqualsTester();
    for (int i = 0; i < instances; i++) {
      tester.addEqualityGroup(generator.generateFresh(type));
    }
    tester.testEquals();
  }

  private static <T> void assertEqualInstance(Class<T> type, T value) {
    FreshValueGenerator generator = new FreshValueGenerator();
