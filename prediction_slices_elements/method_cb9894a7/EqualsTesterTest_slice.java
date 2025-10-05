// Source-based slice around line 260
// Method: <com.google.common.testing.EqualsTesterTest: void testEqualityGroups()>

      tester.testEquals();
    } catch (AssertionFailedError e) {
      assertErrorMessage(
          e, "bar [group 1, item 2] must not be Object#equals to x [group 2, item 2]");
      return;
    }
    fail("should failed because transitivity is broken");
  }

  public void testEqualityGroups() {
    new EqualsTester()
        .addEqualityGroup(named("foo").addPeers("bar"), named("bar").addPeers("foo"))
        .addEqualityGroup(named("baz"), named("baz"))
        .testEquals();
  }

  public void testEqualityBasedOnToString() {
    AssertionFailedError e =
        assertThrows(
            AssertionFailedError.class,
