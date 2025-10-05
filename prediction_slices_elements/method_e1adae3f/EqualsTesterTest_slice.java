// Source-based slice around line 206
// Method: <com.google.common.testing.EqualsTesterTest: void testSymmetryBroken()>

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
      tester.testEquals();
    } catch (AssertionFailedError e) {
      assertErrorMessage(e, "bar [group 1, item 2] must be Object#equals to foo [group 1, item 1]");
      return;
    }
    fail("should failed because symmetry is broken");
  }
