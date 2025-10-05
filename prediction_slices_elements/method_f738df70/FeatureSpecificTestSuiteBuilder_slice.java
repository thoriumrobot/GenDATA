// Source-based slice around line 86
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder: B withSetUp(Runnable)>

    this.subjectGenerator = subjectGenerator;
    return self();
  }

  public G getSubjectGenerator() {
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
