// Source-based slice around line 54
// Method: <com.google.common.collect.testing.SetTestSuiteBuilder: List getTesters()>

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
    testers.add(SetAddTester.class);
    testers.add(SetCreationTester.class);
    testers.add(SetHashCodeTester.class);
    testers.add(SetEqualsTester.class);
    testers.add(SetRemoveTester.class);
    // SetRemoveAllTester doesn't exist because, Sets not permitting
