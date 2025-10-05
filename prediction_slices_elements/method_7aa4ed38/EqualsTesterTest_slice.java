// Source-based slice around line 267
// Method: <com.google.common.testing.EqualsTesterTest: void testEqualityBasedOnToString()>

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
            () ->
                new EqualsTester().addEqualityGroup(new EqualsBasedOnToString("foo")).testEquals());
    assertThat(e).hasMessageThat().contains("toString representation");
  }

  private static void assertErrorMessage(Throwable e, String message) {
    // TODO(kevinb): use a Truth assertion here
