// Source-based slice around line 199
// Method: <com.google.common.testing.EqualsTesterTest: void testNullObjectInEqualityGroup()>

    }
    fail("Should get invalid hashCode error");
  }

  public void testNullEqualityGroup() {
    EqualsTester tester = new EqualsTester();
    assertThrows(NullPointerException.class, () -> tester.addEqualityGroup((Object[]) null));
  }

  public void testNullObjectInEqualityGroup() {
    EqualsTester tester = new EqualsTester();
    NullPointerException e =
        assertThrows(NullPointerException.class, () -> tester.addEqualityGroup(1, null, 3));
    assertErrorMessage(e, "at index 1");
  }

  public void testSymmetryBroken() {
    EqualsTester tester =
        new EqualsTester().addEqualityGroup(named("foo").addPeers("bar"), named("bar"));
    try {
