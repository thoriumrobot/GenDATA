// Source-based slice around line 218
// Method: <com.google.common.testing.EqualsTesterTest: void testTransitivityBrokenInEqualityGroup()>

    try {
      tester.testEquals();
    } catch (AssertionFailedError e) {
      assertErrorMessage(e, "bar [group 1, item 2] must be Object#equals to foo [group 1, item 1]");
      return;
    }
    fail("should failed because symmetry is broken");
  }

  public void testTransitivityBrokenInEqualityGroup() {
    EqualsTester tester =
        new EqualsTester()
            .addEqualityGroup(
                named("foo").addPeers("bar", "baz"),
                named("bar").addPeers("foo"),
                named("baz").addPeers("foo"));
    try {
      tester.testEquals();
    } catch (AssertionFailedError e) {
      assertErrorMessage(e, "bar [group 1, item 2] must be Object#equals to baz [group 1, item 3]");
