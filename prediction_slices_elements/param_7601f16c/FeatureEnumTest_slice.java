// Source-based slice around line 73
// Method: <com.google.common.collect.testing.features.FeatureEnumTest: void assertGoodFeatureEnum(Class)>

              annotationClass, propertyName, annotationClass.getDeclaringClass()),
          annotationClass.getDeclaringClass(),
          returnType.getComponentType());
    }
  }

  // This is public so that tests for Feature enums we haven't yet imagined
  // can reuse it.
  public static <E extends Enum<?> & Feature<?>> void assertGoodFeatureEnum(
      Class<E> featureEnumClass) {
    Class<?>[] classes = featureEnumClass.getDeclaredClasses();
    for (Class<?> containedClass : classes) {
      if (containedClass.getSimpleName().equals("Require")) {
        if (containedClass.isAnnotation()) {
          assertGoodTesterAnnotation(asAnnotation(containedClass));
        } else {
          fail(
              rootLocaleFormat(
                  "Feature enum %s contains a class named "
                      + "'Require' but it is not an annotation.",
