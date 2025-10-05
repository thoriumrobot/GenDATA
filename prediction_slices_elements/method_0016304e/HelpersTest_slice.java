// Source-based slice around line 118
// Method: <com.google.common.collect.testing.HelpersTest: void testAssertContentsInOrder()>


    List<?> differentContents = asList("a", "b", "C");
    try {
      assertEqualInOrder(list, differentContents);
      throw new Error();
    } catch (AssertionFailedError expected) {
    }
  }

  public void testAssertContentsInOrder() {
    List<?> list = asList("a", "b", "c");
    assertContentsInOrder(list, "a", "b", "c");

    try {
      assertContentsInOrder(list, "a", "b");
      throw new Error();
    } catch (AssertionFailedError expected) {
    }

    try {
