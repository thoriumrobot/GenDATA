// Source-based slice around line 71
// Method: <com.google.common.testing.EqualsTesterTest: void testAddNullEqualObject()>


  /** Test equalObjects after adding multiple instances at once with a null */
  public void testAddTwoEqualObjectsAtOnceWithNull() {
    assertThrows(
        NullPointerException.class,
        () -> equalsTester.addEqualityGroup(reference, equalObject1, null));
  }

  /** Test adding null equal object yields error */
  public void testAddNullEqualObject() {
    assertThrows(
        NullPointerException.class,
        () -> equalsTester.addEqualityGroup(reference, (Object[]) null));
  }

  /**
   * Test adding objects only by addEqualityGroup, with no reference object specified in the
   * constructor.
   */
  public void testAddEqualObjectWithOArgConstructor() {
