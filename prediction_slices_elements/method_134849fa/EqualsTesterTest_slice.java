// Source-based slice around line 245
// Method: <com.google.common.testing.EqualsTesterTest: void testTransitivityBrokenAcrossEqualityGroups()>

    try {
      tester.testEquals();
    } catch (AssertionFailedError e) {
      assertErrorMessage(e, "foo [group 1, item 1] must be Object#equals to bar [group 1, item 2]");
      return;
    }
    fail("should failed because of unequal objects in the same equality group");
  }

  public void testTransitivityBrokenAcrossEqualityGroups() {
    EqualsTester tester =
        new EqualsTester()
            .addEqualityGroup(named("foo").addPeers("bar"), named("bar").addPeers("foo", "x"))
            .addEqualityGroup(named("baz").addPeers("x"), named("x").addPeers("baz", "bar"));
    try {
      tester.testEquals();
    } catch (AssertionFailedError e) {
      assertErrorMessage(
          e, "bar [group 1, item 2] must not be Object#equals to x [group 2, item 2]");
      return;
