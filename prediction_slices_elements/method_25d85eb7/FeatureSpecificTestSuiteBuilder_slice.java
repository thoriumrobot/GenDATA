// Source-based slice around line 308
// Method: <com.google.common.collect.testing.FeatureSpecificTestSuiteBuilder: String formatFeatureSet(Set)>

    while (tests.hasMoreElements()) {
      Test test = (Test) tests.nextElement();
      if (matches(test)) {
        filtered.addTest(test);
      }
    }
    return filtered;
  }

  protected static String formatFeatureSet(Set<? extends Feature<?>> features) {
    List<String> temp = new ArrayList<>();
    for (Feature<?> feature : features) {
      Object featureAsObject = feature; // to work around bogus JDK warning
      if (featureAsObject instanceof Enum) {
        Enum<?> f = (Enum<?>) featureAsObject;
        temp.add(f.getDeclaringClass().getSimpleName() + "." + feature);
      } else {
        temp.add(feature.toString());
      }
    }
