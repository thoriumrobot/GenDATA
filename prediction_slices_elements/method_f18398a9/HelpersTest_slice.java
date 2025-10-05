// Source-based slice around line 159
// Method: <com.google.common.collect.testing.HelpersTest: void testAssertContainsAllOf()>

    assertContains(list, "b");

    try {
      assertContains(list, "c");
      throw new Error();
    } catch (AssertionFailedError expected) {
    }
  }

  public void testAssertContainsAllOf() {
    List<?> list = asList("a", "a", "b", "c");
    assertContainsAllOf(list, "a");
    assertContainsAllOf(list, "a", "a");
    assertContainsAllOf(list, "a", "b", "c");
    assertContainsAllOf(list, "a", "b", "c", "a");

    try {
      assertContainsAllOf(list, "d");
      throw new Error();
    } catch (AssertionFailedError expected) {
