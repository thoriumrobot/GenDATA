// Source-based slice around line 568
// Method: <com.google.common.testing.anotherpackage.ForwardingWrapperTesterTest: void testExplicitEqualsAndHashCodeNotDelegatedByDefault()>

      this.delegate = delegate;
    }

    @Override
    public String toString() {
      return delegate.toString();
    }
  }

  public void testExplicitEqualsAndHashCodeNotDelegatedByDefault() {
    new ForwardingWrapperTester().testForwarding(Equals.class, NoDelegateToEquals.WRAPPER);
  }

  public void testExplicitEqualsAndHashCodeDelegatedWhenExplicitlyAsked() {
    try {
      new ForwardingWrapperTester()
          .includingEquals()
          .testForwarding(Equals.class, NoDelegateToEquals.WRAPPER);
    } catch (AssertionFailedError expected) {
      return;
