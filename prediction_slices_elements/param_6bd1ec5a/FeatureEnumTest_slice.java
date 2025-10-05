// Source-based slice around line 111
// Method: <com.google.common.collect.testing.features.FeatureEnumTest: String rootLocaleFormat(String,Object)>


  public void testFeatureEnums() throws Exception {
    assertGoodFeatureEnum(CollectionFeature.class);
    assertGoodFeatureEnum(ListFeature.class);
    assertGoodFeatureEnum(SetFeature.class);
    assertGoodFeatureEnum(CollectionSize.class);
    assertGoodFeatureEnum(MapFeature.class);
  }

  private static String rootLocaleFormat(String format, Object... args) {
    return String.format(Locale.ROOT, format, args);
  }
}
