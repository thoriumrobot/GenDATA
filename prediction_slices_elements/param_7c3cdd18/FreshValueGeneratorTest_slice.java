// Source-based slice around line 539
// Method: <com.google.common.testing.FreshValueGeneratorTest: void assertValueAndTypeEquals(Object,Object)>

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
