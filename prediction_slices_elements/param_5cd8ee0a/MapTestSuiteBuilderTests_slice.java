// Source-based slice around line 80
// Method: <com.google.common.collect.testing.MapTestSuiteBuilderTests: TestSuite wrappedHashMapTests(WrappedHashMapGenerator,String,Feature)>

        map.put(entry.getKey(), entry.getValue());
      }
      return wrap(map);
    }

    abstract Map<String, String> wrap(HashMap<String, String> map);
  }

  private static TestSuite wrappedHashMapTests(
      WrappedHashMapGenerator generator, String name, Feature<?>... features) {
    List<Feature<?>> featuresList = Lists.newArrayList(features);
    Collections.addAll(
        featuresList,
        MapFeature.GENERAL_PURPOSE,
        CollectionFeature.SUPPORTS_ITERATOR_REMOVE,
        CollectionSize.ANY);
    return MapTestSuiteBuilder.using(generator)
        .named(name)
        .withFeatures(featuresList)
        .createTestSuite();
