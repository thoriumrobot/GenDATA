// Source-based slice around line 495
// Method: <com.google.common.testing.FreshValueGeneratorTest: void assertFreshInstance(TypeToken)>

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
    assertFreshInstance(TypeToken.of(type), instances);
  }

  private static void assertFreshInstance(TypeToken<?> type, int instances) {
    FreshValueGenerator generator = new FreshValueGenerator();
    EqualsTester tester = new EqualsTester();
