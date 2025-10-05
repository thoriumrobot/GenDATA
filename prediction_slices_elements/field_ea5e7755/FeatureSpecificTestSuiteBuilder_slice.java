// Source-based slice around line 71
// Method: com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder.setUp

  @SuppressWarnings("unchecked")
  protected B self() {
    return (B) this;
  }

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
