// Source-based slice around line 48
// Method: <com.google.common.collect.testing.SetTestSuiteBuilder: SetTestSuiteBuilder using(TestSetGenerator)>


/**
 * Creates, based on your criteria, a JUnit test suite that exhaustively tests a Set implementation.
 *
 * @author George van den Driessche
 */
@GwtIncompatible
public class SetTestSuiteBuilder<E>
    extends AbstractCollectionTestSuiteBuilder<SetTestSuiteBuilder<E>, E> {
  public static <E> SetTestSuiteBuilder<E> using(TestSetGenerator<E> generator) {
    return new SetTestSuiteBuilder<E>().usingGenerator(generator);
  }

  @SuppressWarnings("rawtypes") // class literals
  @Override
  protected List<Class<? extends AbstractTester>> getTesters() {
    List<Class<? extends AbstractTester>> testers = copyToList(super.getTesters());

    testers.add(CollectionSerializationEqualTester.class);
    testers.add(SetAddAllTester.class);
