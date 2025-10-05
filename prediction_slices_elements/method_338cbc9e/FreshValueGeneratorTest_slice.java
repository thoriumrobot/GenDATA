// Source-based slice around line 202
// Method: <com.google.common.testing.FreshValueGeneratorTest: void testPrimitiveArray()>

  }

  public void testStringArray() {
    FreshValueGenerator generator = new FreshValueGenerator();
    String[] a1 = generator.generateFresh(String[].class);
    String[] a2 = generator.generateFresh(String[].class);
    assertFalse(a1[0].equals(a2[0]));
  }

  public void testPrimitiveArray() {
    FreshValueGenerator generator = new FreshValueGenerator();
    int[] a1 = generator.generateFresh(int[].class);
    int[] a2 = generator.generateFresh(int[].class);
    assertTrue(a1[0] != a2[0]);
  }

  public void testRange() {
    assertFreshInstance(new TypeToken<Range<String>>() {});
  }

