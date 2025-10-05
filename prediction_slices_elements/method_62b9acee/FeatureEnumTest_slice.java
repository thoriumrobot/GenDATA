// Source-based slice around line 95
// Method: <com.google.common.collect.testing.features.FeatureEnumTest: Class asAnnotation(Class)>

        return;
      }
    }
    fail(
        rootLocaleFormat(
            "Feature enum %s should contain an annotation named 'Require'.", featureEnumClass));
  }

  @SuppressWarnings("unchecked")
  private static Class<? extends Annotation> asAnnotation(Class<?> clazz) {
    if (clazz.isAnnotation()) {
      return (Class<? extends Annotation>) clazz;
    } else {
      throw new IllegalArgumentException(rootLocaleFormat("%s is not an annotation.", clazz));
    }
  }

  public void testFeatureEnums() throws Exception {
    assertGoodFeatureEnum(CollectionFeature.class);
    assertGoodFeatureEnum(ListFeature.class);
