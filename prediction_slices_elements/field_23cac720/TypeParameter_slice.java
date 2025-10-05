// Source-based slice around line 48
// Method: com.google.common.reflect.TypeParameter.typeVariable

 * wouldn't behave as users might expect. Additionally, it's not clear how the TypeToken API could
 * support even a "normal" `TypeParameter<T>` when `<T>` has a nullable bound. (See the discussion
 * on TypeToken.where.) So, in the interest of failing fast and encouraging the user to switch to a
 * non-null bound if possible, let's require a non-null bound here.
 *
 * TODO(cpovirk): Elaborate on "wouldn't behave as users might expect."
 */
public abstract class TypeParameter<T> extends TypeCapture<T> {

  final TypeVariable<?> typeVariable;

  protected TypeParameter() {
    Type type = capture();
    checkArgument(type instanceof TypeVariable, "%s should be a type variable.", type);
    this.typeVariable = (TypeVariable<?>) type;
  }

  @Override
  public final int hashCode() {
    return typeVariable.hashCode();
