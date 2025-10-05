// Source-based slice around line 107
// Method: com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder.features

    return self();
  }

  public Runnable getTearDown() {
    return tearDown;
  }

  // Features

  private final Set<Feature<?>> features = new LinkedHashSet<>();

  /**
   * Configures this builder to produce tests appropriate for the given features. This method may be
   * called more than once to add features in multiple groups.
   */
  @CanIgnoreReturnValue
  public B withFeatures(Feature<?>... features) {
    return withFeatures(asList(features));
  }

