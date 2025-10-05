// Source-based slice around line 116
// Method: <com.google.common.testing.EqualsTesterTest: void testNonReflexiveEquals()>

   * Test EqualsTester after populating equalObjects. This checks proper handling of equality and
   * verifies hashCode for valid objects
   */
  public void testTestEqualsEqualsObjects() {
    equalsTester.addEqualityGroup(reference, equalObject1, equalObject2);
    equalsTester.testEquals();
  }

  /** Test proper handling of case where an object is not equal to itself */
  public void testNonReflexiveEquals() {
    Object obj = new NonReflexiveObject();
    equalsTester.addEqualityGroup(obj);
    try {
      equalsTester.testEquals();
    } catch (AssertionFailedError e) {
      assertErrorMessage(e, obj + " must be Object#equals to itself");
      return;
    }
    fail("Should get non-reflexive error");
  }
