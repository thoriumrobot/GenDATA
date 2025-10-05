// Source-based slice around line 282
// Method: <com.google.common.testing.anotherpackage.ForwardingWrapperTesterTest: void assertFailure(Class,Function,String)>

                .testForwarding(String.class, Functions.<String>identity()));
  }

  public void testNulls() {
    new NullPointerTester()
        .setDefault(Class.class, Runnable.class)
        .testAllPublicInstanceMethods(new ForwardingWrapperTester());
  }

  private <T> void assertFailure(
      Class<T> interfaceType,
      Function<T, ? extends T> wrapperFunction,
      String... expectedMessages) {
    try {
      tester.testForwarding(interfaceType, wrapperFunction);
    } catch (AssertionFailedError expected) {
      for (String message : expectedMessages) {
        assertThat(expected).hasMessageThat().contains(message);
      }
      return;
