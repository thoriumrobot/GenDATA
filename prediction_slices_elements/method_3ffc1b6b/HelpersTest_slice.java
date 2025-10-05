// Source-based slice around line 86
// Method: <com.google.common.collect.testing.HelpersTest: void testAssertEqualInOrder()>


    map.put("a", "b");
    try {
      assertEmpty(map);
      throw new Error();
    } catch (AssertionFailedError expected) {
    }
  }

  public void testAssertEqualInOrder() {
    List<?> list = asList("a", "b", "c");
    assertEqualInOrder(list, list);

    List<?> fewer = asList("a", "b");
    try {
      assertEqualInOrder(list, fewer);
      throw new Error();
    } catch (AssertionFailedError expected) {
    }

