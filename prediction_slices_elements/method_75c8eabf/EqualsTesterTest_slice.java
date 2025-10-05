// Source-based slice around line 234
// Method: <com.google.common.testing.EqualsTesterTest: void testUnequalObjectsInEqualityGroup()>

    try {
      tester.testEquals();
    } catch (AssertionFailedError e) {
      assertErrorMessage(e, "bar [group 1, item 2] must be Object#equals to baz [group 1, item 3]");
      return;
    }
    fail("should failed because transitivity is broken");
  }

  public void testUnequalObjectsInEqualityGroup() {
    EqualsTester tester = new EqualsTester().addEqualityGroup(named("foo"), named("bar"));
    try {
      tester.testEquals();
    } catch (AssertionFailedError e) {
      assertErrorMessage(e, "foo [group 1, item 1] must be Object#equals to bar [group 1, item 2]");
      return;
    }
    fail("should failed because of unequal objects in the same equality group");
  }

