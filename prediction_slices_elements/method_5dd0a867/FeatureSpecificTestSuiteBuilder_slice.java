// Source-based slice around line 101
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder: Runnable getTearDown()>

    return setUp;
  }

  @CanIgnoreReturnValue
  public B withTearDown(Runnable tearDown) {
    this.tearDown = tearDown;
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
