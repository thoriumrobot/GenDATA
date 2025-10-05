// Source-based slice around line 276
// Method: <com.google.common.testing.anotherpackage.ForwardingWrapperTesterTest: void testNulls()>


  public void testNotInterfaceType() {
    assertThrows(
        IllegalArgumentException.class,
        () ->
            new ForwardingWrapperTester()
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
