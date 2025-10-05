// Source-based slice around line 30
// Method: <com.google.common.reflect.TypeCapture: Type capture()>


/**
 * Captures the actual type of {@code T}.
 *
 * @author Ben Yu
 */
abstract class TypeCapture<T> {

  /** Returns the captured type. */
  final Type capture() {
    Type superclass = getClass().getGenericSuperclass();
    checkArgument(superclass instanceof ParameterizedType, "%s isn't parameterized", superclass);
    return ((ParameterizedType) superclass).getActualTypeArguments()[0];
  }
}
