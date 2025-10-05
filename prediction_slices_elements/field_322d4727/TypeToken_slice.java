// Source-based slice around line 107
// Method: com.google.common.reflect.TypeToken.invariantTypeResolver

 * @author Ben Yu
 * @since 12.0
 */
@SuppressWarnings("serial") // SimpleTypeToken is the serialized form.
public abstract class TypeToken<T> extends TypeCapture<T> implements Serializable {

  private final Type runtimeType;

  /** Resolver for resolving parameter and field types with {@link #runtimeType} as context. */
  @LazyInit private transient @Nullable TypeResolver invariantTypeResolver;

  /** Resolver for resolving covariant types with {@link #runtimeType} as context. */
  @LazyInit private transient @Nullable TypeResolver covariantTypeResolver;

  /**
   * Constructs a new type token of {@code T}.
   *
   * <p>Clients create an empty anonymous subclass. Doing so embeds the type parameter in the
   * anonymous class's type hierarchy so we can reconstitute it at runtime despite erasure.
   *
