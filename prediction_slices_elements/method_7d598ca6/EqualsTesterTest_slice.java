// Source-based slice around line 64
// Method: <com.google.common.testing.EqualsTesterTest: void testAddTwoEqualObjectsAtOnceWithNull()>

    notEqualObject1 = new ValidTestObject(0, 2);
  }

  /** Test null reference yields error */
  public void testAddNullReference() {
    assertThrows(NullPointerException.class, () -> equalsTester.addEqualityGroup((Object) null));
  }

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
