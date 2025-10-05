// Source-based slice around line 81
// Method: <com.google.common.testing.EqualsTesterTest: void testAddEqualObjectWithOArgConstructor()>

    assertThrows(
        NullPointerException.class,
        () -> equalsTester.addEqualityGroup(reference, (Object[]) null));
  }

  /**
   * Test adding objects only by addEqualityGroup, with no reference object specified in the
   * constructor.
   */
  public void testAddEqualObjectWithOArgConstructor() {
    equalsTester.addEqualityGroup(equalObject1, notEqualObject1);
    try {
      equalsTester.testEquals();
    } catch (AssertionFailedError e) {
      assertErrorMessage(
          e,
          equalObject1
              + " [group 1, item 1] must be Object#equals to "
              + notEqualObject1
              + " [group 1, item 2]");
