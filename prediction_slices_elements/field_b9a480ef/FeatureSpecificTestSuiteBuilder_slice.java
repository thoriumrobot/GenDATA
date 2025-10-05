// Source-based slice around line 69
// Method: com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder.subjectGenerator

public abstract class FeatureSpecificTestSuiteBuilder<
    B extends FeatureSpecificTestSuiteBuilder<B, G>, G> {
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
