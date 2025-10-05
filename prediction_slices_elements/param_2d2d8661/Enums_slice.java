// Source-based slice around line 49
// Method: <com.google.common.base.Enums: Field getField(Enum)>

  private Enums() {}

  /**
   * Returns the {@link Field} in which {@code enumValue} is defined. For example, to get the {@code
   * Description} annotation on the {@code GOLF} constant of enum {@code Sport}, use {@code
   * Enums.getField(Sport.GOLF).getAnnotation(Description.class)}.
   *
   * @since 12.0
   */
  public static Field getField(Enum<?> enumValue) {
    Class<?>
        clazz = enumValue.getDeclaringClass();
    try {
      return clazz.getDeclaredField(enumValue.name());
    } catch (NoSuchFieldException impossible) {
      throw new AssertionError(impossible);
    }
  }

  /**
