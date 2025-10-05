// Source-based slice around line 86
// Method: <com.google.common.base.Enums: Map getEnumConstants(Class)>

      Class<T> enumClass) {
    Map<String, WeakReference<? extends Enum<?>>> result = new HashMap<>();
    for (T enumInstance : EnumSet.allOf(enumClass)) {
      result.put(enumInstance.name(), new WeakReference<Enum<?>>(enumInstance));
    }
    enumConstantCache.put(enumClass, result);
    return result;
  }

  static <T extends Enum<T>> Map<String, WeakReference<? extends Enum<?>>> getEnumConstants(
      Class<T> enumClass) {
    synchronized (enumConstantCache) {
      Map<String, WeakReference<? extends Enum<?>>> constants = enumConstantCache.get(enumClass);
      if (constants == null) {
        constants = populateCache(enumClass);
      }
      return constants;
    }
  }

