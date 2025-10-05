// Source-based slice around line 147
// Method: <com.google.common.collect.testing.HelpersTest: void testAssertContains()>

    }

    try {
      assertContentsInOrder(list, "a", "B", "c");
      throw new Error();
    } catch (AssertionFailedError expected) {
    }
  }

  public void testAssertContains() {
    List<?> list = asList("a", "b");
    assertContains(list, "a");
    assertContains(list, "b");

    try {
      assertContains(list, "c");
      throw new Error();
    } catch (AssertionFailedError expected) {
    }
  }
