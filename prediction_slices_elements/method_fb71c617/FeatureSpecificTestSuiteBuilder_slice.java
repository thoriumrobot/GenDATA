// Source-based slice around line 96
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder: B withTearDown(Runnable)>

    this.setUp = setUp;
    return self();
  }

  public Runnable getSetUp() {
    return setUp;
  }

  @CanIgnoreReturnValue
  public B withTearDown(Runnable tearDown) {
    this.tearDown = tearDown;
    return self();
  }

  public Runnable getTearDown() {
    return tearDown;
  }

  // Features

