// Source-based slice around line 67
// Method: <com.google.common.base.Enums: Optional getIfPresent(Class,String)>


  /**
   * Returns an optional enum constant for the given type, using {@link Enum#valueOf}. If the
   * constant does not exist, {@link Optional#absent} is returned. A common use case is for parsing
   * user input or falling back to a default enum constant. For example, {@code
   * Enums.getIfPresent(Country.class, countryInput).or(Country.DEFAULT);}
   *
   * @since 12.0
   */
  public static <T extends Enum<T>> Optional<T> getIfPresent(Class<T> enumClass, String value) {
    checkNotNull(enumClass);
    checkNotNull(value);
    return Platform.getEnumIfPresent(enumClass, value);
  }

  private static final Map<Class<? extends Enum<?>>, Map<String, WeakReference<? extends Enum<?>>>>
      enumConstantCache = new WeakHashMap<>();

  private static <T extends Enum<T>> Map<String, WeakReference<? extends Enum<?>>> populateCache(
      Class<T> enumClass) {
