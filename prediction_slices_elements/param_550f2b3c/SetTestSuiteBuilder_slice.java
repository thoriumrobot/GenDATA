// Source-based slice around line 118
// Method: <com.google.common.collect.testing.SetTestSuiteBuilder: Set computeReserializedCollectionFeatures(Set)>

      return gen.createArray(length);
    }

    @Override
    public Iterable<E> order(List<E> insertionOrder) {
      return gen.order(insertionOrder);
    }
  }

  private static Set<Feature<?>> computeReserializedCollectionFeatures(Set<Feature<?>> features) {
    Set<Feature<?>> derivedFeatures = new HashSet<>(features);
    derivedFeatures.remove(SERIALIZABLE);
    derivedFeatures.remove(SERIALIZABLE_INCLUDING_VIEWS);
    return derivedFeatures;
  }
}
