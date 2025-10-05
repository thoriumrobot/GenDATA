// Source-based slice around line 499
// Method: <com.google.common.testing.FreshValueGeneratorTest: void assertFreshInstance(Class,int)>

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
    for (int i = 0; i < instances; i++) {
      tester.addEqualityGroup(generator.generateFresh(type));
    }
    tester.testEquals();
