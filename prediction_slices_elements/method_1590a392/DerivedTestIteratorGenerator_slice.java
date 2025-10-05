// Source-based slice around line 38
// Method: <com.google.common.collect.testing.DerivedTestIteratorGenerator: TestSubjectGenerator getInnerGenerator()>

    implements TestIteratorGenerator<E>, DerivedGenerator {
  private final TestSubjectGenerator<? extends Iterable<E>> collectionGenerator;

  public DerivedTestIteratorGenerator(
      TestSubjectGenerator<? extends Iterable<E>> collectionGenerator) {
    this.collectionGenerator = collectionGenerator;
  }

  @Override
  public TestSubjectGenerator<? extends Iterable<E>> getInnerGenerator() {
    return collectionGenerator;
  }

  @Override
  public Iterator<E> get() {
    return collectionGenerator.createTestSubject().iterator();
  }
}
