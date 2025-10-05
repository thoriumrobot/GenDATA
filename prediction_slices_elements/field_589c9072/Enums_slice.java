// Source-based slice around line 73
// Method: com.google.common.base.Enums.enumConstantCache

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
    Map<String, WeakReference<? extends Enum<?>>> result = new HashMap<>();
    for (T enumInstance : EnumSet.allOf(enumClass)) {
      result.put(enumInstance.name(), new WeakReference<Enum<?>>(enumInstance));
    }
    enumConstantCache.put(enumClass, result);
    return result;
