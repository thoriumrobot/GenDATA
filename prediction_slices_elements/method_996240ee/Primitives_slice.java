// Source-based slice around line 115
// Method: <com.google.common.primitives.Primitives: Class wrap(Class)>

   * Returns the corresponding wrapper type of {@code type} if it is a primitive type; otherwise
   * returns {@code type} itself. Idempotent.
   *
   * <pre>
   *     wrap(int.class) == Integer.class
   *     wrap(Integer.class) == Integer.class
   *     wrap(String.class) == String.class
   * </pre>
   */
  public static <T> Class<T> wrap(Class<T> type) {
    checkNotNull(type);

    // cast is safe: long.class and Long.class are both of type Class<Long>
    @SuppressWarnings("unchecked")
    Class<T> wrapped = (Class<T>) PRIMITIVE_TO_WRAPPER_TYPE.get(type);
    return (wrapped == null) ? type : wrapped;
  }

  /**
   * Returns the corresponding primitive type of {@code type} if it is a wrapper type; otherwise
