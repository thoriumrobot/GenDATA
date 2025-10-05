// Source-based slice around line 142
// Method: <com.google.common.testing.EqualsTesterTest: void testInvalidEqualsIncompatibleClass()>

      equalsTester.testEquals();
    } catch (AssertionFailedError e) {
      assertErrorMessage(e, obj + " must not be Object#equals to null");
      return;
    }
    fail("Should get equal to null error");
  }

  /** Test proper handling where an object incorrectly tests for an incompatible class */
  public void testInvalidEqualsIncompatibleClass() {
    Object obj = new InvalidEqualsIncompatibleClassObject();
    equalsTester.addEqualityGroup(obj);
    try {
      equalsTester.testEquals();
    } catch (AssertionFailedError e) {
      assertErrorMessage(
          e, obj + " must not be Object#equals to an arbitrary object of another class");
      return;
    }
    fail("Should get equal to incompatible class error");
