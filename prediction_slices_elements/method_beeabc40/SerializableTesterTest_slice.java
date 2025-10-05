// Source-based slice around line 66
// Method: <com.google.common.testing.SerializableTesterTest: void testObjectWhichIsEqualButChangesClass()>

      SerializableTester.reserializeAndAssert(orig);
      errorNotThrown = true;
    } catch (AssertionFailedError error) {
      // expected
      assertContains("must be equal to the Object#hashCode", error.getMessage());
    }
    assertFalse(errorNotThrown);
  }

  public void testObjectWhichIsEqualButChangesClass() {
    ObjectWhichIsEqualButChangesClass orig = new ObjectWhichIsEqualButChangesClass();
    boolean errorNotThrown = false;
    try {
      SerializableTester.reserializeAndAssert(orig);
      errorNotThrown = true;
    } catch (AssertionFailedError error) {
      // expected
      assertContains("expected:<class ", error.getMessage());
    }
    assertFalse(errorNotThrown);
