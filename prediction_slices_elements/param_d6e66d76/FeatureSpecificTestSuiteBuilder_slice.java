// Source-based slice around line 119
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder: B withFeatures(Iterable)>

   * Configures this builder to produce tests appropriate for the given features. This method may be
   * called more than once to add features in multiple groups.
   */
  @CanIgnoreReturnValue
  public B withFeatures(Feature<?>... features) {
    return withFeatures(asList(features));
  }

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

