// Source-based slice around line 71
// Method: <com.google.common.collect.testing.SetTestSuiteBuilder: List createDerivedSuites(FeatureSpecificTestSuiteBuilder)>

    testers.add(SetEqualsTester.class);
    testers.add(SetRemoveTester.class);
    // SetRemoveAllTester doesn't exist because, Sets not permitting
    // duplicate elements, there are no tests for Set.removeAll() that aren't
    // covered by CollectionRemoveAllTester.
    return testers;
  }

  @Override
  protected List<TestSuite> createDerivedSuites(
      FeatureSpecificTestSuiteBuilder<?, ? extends OneSizeTestContainerGenerator<Collection<E>, E>>
          parentBuilder) {
    List<TestSuite> derivedSuites = new ArrayList<>(super.createDerivedSuites(parentBuilder));

    if (parentBuilder.getFeatures().contains(SERIALIZABLE)) {
      derivedSuites.add(
          SetTestSuiteBuilder.using(
                  new ReserializedSetGenerator<E>(parentBuilder.getSubjectGenerator()))
              .named(getName() + " reserialized")
              .withFeatures(computeReserializedCollectionFeatures(parentBuilder.getFeatures()))
