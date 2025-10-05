// Source-based slice around line 529
// Method: <com.google.common.testing.FreshValueGeneratorTest: void assertCanGenerateOnly(TypeToken,Object)>

  private enum OneConstantEnum {
    CONSTANT1
  }

  private enum TwoConstantEnum {
    CONSTANT1,
    CONSTANT2
  }

  private static void assertCanGenerateOnly(TypeToken<?> type, Object expected) {
    FreshValueGenerator generator = new FreshValueGenerator();
    assertValueAndTypeEquals(expected, generator.generateFresh(type));
    assertNull(generator.generateFresh(type));
  }

  private static void assertNotInstantiable(TypeToken<?> type) {
    assertNull(new FreshValueGenerator().generateFresh(type));
  }

  private static void assertValueAndTypeEquals(Object expected, Object actual) {
