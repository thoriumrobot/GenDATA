// Source-based slice around line 44
// Method: com.google.common.primitives.Primitives.WRAPPER_TO_PRIMITIVE_TYPE


  /** A map from primitive types to their corresponding wrapper types. */
  // It's a constant, and we can't use ImmutableMap here without creating a circular dependency.
  @SuppressWarnings("ConstantCaseForConstants")
  private static final Map<Class<?>, Class<?>> PRIMITIVE_TO_WRAPPER_TYPE;

  /** A map from wrapper types to their corresponding primitive types. */
  // It's a constant, and we can't use ImmutableMap here without creating a circular dependency.
  @SuppressWarnings("ConstantCaseForConstants")
  private static final Map<Class<?>, Class<?>> WRAPPER_TO_PRIMITIVE_TYPE;

  // Sad that we can't use a BiMap. :(

  static {
    Map<Class<?>, Class<?>> primToWrap = new LinkedHashMap<>(16);
    Map<Class<?>, Class<?>> wrapToPrim = new LinkedHashMap<>(16);

    add(primToWrap, wrapToPrim, boolean.class, Boolean.class);
    add(primToWrap, wrapToPrim, byte.class, Byte.class);
    add(primToWrap, wrapToPrim, char.class, Character.class);
