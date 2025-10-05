// Source-based slice around line 535
// Method: <com.google.common.testing.FreshValueGeneratorTest: void assertNotInstantiable(TypeToken)>

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
    assertEquals(expected, actual);
    assertEquals(expected.getClass(), actual.getClass());
  }
}
