// Source-based slice around line 110
// Method: <com.google.common.testing.EqualsTesterTest: void testTestEqualsEqualsObjects()>

  public void testTestEqualsEmptyLists() {
    equalsTester.addEqualityGroup(reference);
    equalsTester.testEquals();
  }

  /**
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
