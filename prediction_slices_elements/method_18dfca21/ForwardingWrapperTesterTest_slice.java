// Source-based slice around line 572
// Method: <com.google.common.testing.anotherpackage.ForwardingWrapperTesterTest: void testExplicitEqualsAndHashCodeDelegatedWhenExplicitlyAsked()>

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
    }
    fail("Should have failed");
  }

