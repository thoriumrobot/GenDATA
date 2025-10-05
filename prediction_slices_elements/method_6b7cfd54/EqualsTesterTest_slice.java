// Source-based slice around line 276
// Method: <com.google.common.testing.EqualsTesterTest: void assertErrorMessage(Throwable,String)>

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
    if (!e.getMessage().contains(message)) {
      fail("expected <" + e.getMessage() + "> to contain <" + message + ">");
    }
  }

  /**
   * Test class with valid equals and hashCode methods. Testers created with instances of this class
   * should always pass.
   */
