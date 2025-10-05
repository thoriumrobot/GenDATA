// Source-based slice around line 39
// Method: com.google.common.primitives.Primitives.PRIMITIVE_TO_WRAPPER_TYPE

 * @since 1.0
 */
@GwtCompatible
public final class Primitives {
  private Primitives() {}

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
