// Source-based slice around line 52
// Method: <com.google.common.testing.SerializableTesterTest: void testClassWhichIsAlwaysEqualButHasDifferentHashcodes()>

      SerializableTester.reserializeAndAssert(orig);
      errorNotThrown = true;
    } catch (AssertionFailedError error) {
      // expected
      assertContains("must be Object#equals to", error.getMessage());
    }
    assertFalse(errorNotThrown);
  }

  public void testClassWhichIsAlwaysEqualButHasDifferentHashcodes() {
    ClassWhichIsAlwaysEqualButHasDifferentHashcodes orig =
        new ClassWhichIsAlwaysEqualButHasDifferentHashcodes();
    boolean errorNotThrown = false;
    try {
      SerializableTester.reserializeAndAssert(orig);
      errorNotThrown = true;
    } catch (AssertionFailedError error) {
      // expected
      assertContains("must be equal to the Object#hashCode", error.getMessage());
    }
