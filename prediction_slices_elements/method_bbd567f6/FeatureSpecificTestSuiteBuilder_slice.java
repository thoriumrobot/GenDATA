// Source-based slice around line 126
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder: Set getFeatures()>


  @CanIgnoreReturnValue
  public B withFeatures(Iterable<? extends Feature<?>> features) {
    for (Feature<?> feature : features) {
      this.features.add(feature);
    }
    return self();
  }

  public Set<Feature<?>> getFeatures() {
    return unmodifiableSet(features);
  }

  // Name

  private @Nullable String name;

  /** Configures this builder produce a TestSuite with the given name. */
  @CanIgnoreReturnValue
  public B named(String name) {
