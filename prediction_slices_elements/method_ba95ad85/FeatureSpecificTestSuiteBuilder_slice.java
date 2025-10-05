// Source-based slice around line 63
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder: B self()>

 * @param <G> The type of the generator to be passed to testers in the generated test suite. An
 *     instance of G should somehow provide an instance of the class under test, plus any other
 *     information required to parameterize the test.
 * @author George van den Driessche
 */
@GwtIncompatible
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
