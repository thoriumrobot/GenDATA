// Source-based slice around line 59
// Method: <com.google.common.testing.EqualsTesterTest: void testAddNullReference()>

    super.setUp();
    reference = new ValidTestObject(1, 2);
    equalsTester = new EqualsTester();
    equalObject1 = new ValidTestObject(1, 2);
    equalObject2 = new ValidTestObject(1, 2);
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

