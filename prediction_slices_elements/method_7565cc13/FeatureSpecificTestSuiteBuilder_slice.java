// Source-based slice around line 91
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder: Runnable getSetUp()>

    return subjectGenerator;
  }

  @CanIgnoreReturnValue
  public B withSetUp(Runnable setUp) {
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
