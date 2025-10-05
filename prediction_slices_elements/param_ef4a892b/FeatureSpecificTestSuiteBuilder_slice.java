// Source-based slice around line 76
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder: B usingGenerator(G)>

  // Test Data

  private @Nullable G subjectGenerator;
  // Gets run before every test.
  private Runnable setUp;
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
