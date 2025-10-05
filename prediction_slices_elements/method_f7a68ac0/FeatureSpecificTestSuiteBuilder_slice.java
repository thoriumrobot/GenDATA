// Source-based slice around line 81
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder: G getSubjectGenerator()>

  // Gets run at the conclusion of every test.
  private Runnable tearDown;

  @CanIgnoreReturnValue
  protected B usingGenerator(G subjectGenerator) {
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
