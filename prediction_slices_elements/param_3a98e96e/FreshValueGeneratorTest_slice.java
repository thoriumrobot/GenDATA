// Source-based slice around line 512
// Method: <com.google.common.testing.FreshValueGeneratorTest: void assertEqualInstance(Class,T)>

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
    assertEquals(value, generator.generateFresh(type));
    assertEquals(value, generator.generateFresh(type));
  }

  private enum EmptyEnum {}

  private enum OneConstantEnum {
    CONSTANT1
  }
