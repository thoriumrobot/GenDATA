// Source-based slice around line 43
// Method: <com.google.common.collect.testing.DerivedTestIteratorGenerator: Iterator get()>

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
