// Source-based slice around line 172
// Method: <com.google.common.testing.EqualsTesterTest: void testInvalidHashCode()>

      return;
    }
    fail("Should get not equal to equal object error");
  }

  /**
   * Test for an invalid hashCode method, i.e., one that returns different value for objects that
   * are equal according to the equals method
   */
  public void testInvalidHashCode() {
    Object a = new InvalidHashCodeObject(1, 2);
    Object b = new InvalidHashCodeObject(1, 2);
    equalsTester.addEqualityGroup(a, b);
    try {
      equalsTester.testEquals();
    } catch (AssertionFailedError e) {
      assertErrorMessage(
          e,
          "the Object#hashCode ("
              + a.hashCode()
