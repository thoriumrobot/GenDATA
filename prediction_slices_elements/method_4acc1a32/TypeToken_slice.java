// Source-based slice around line 210
// Method: <com.google.common.reflect.TypeToken: Type getType()>

      return result;
    } else {
      // For a wildcard or type variable, the first bound determines the runtime type.
      // This case also covers GenericArrayType.
      return getRawTypes().iterator().next();
    }
  }

  /** Returns the represented type. */
  public final Type getType() {
    return runtimeType;
  }

  /**
   * Returns a new {@code TypeToken} where type variables represented by {@code typeParam} are
   * substituted by {@code typeArg}. For example, it can be used to construct {@code Map<K, V>} for
   * any {@code K} and {@code V} type:
   *
   * {@snippet :
   * static <K, V> TypeToken<Map<K, V>> mapOf(
