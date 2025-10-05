// Source-based slice around line 1067
// Method: <com.google.common.reflect.TypeToken: Bounds any(Type[])>

    }
    return Types.newParameterizedTypeWithOwner(type.getOwnerType(), rawType, typeArgs);
  }

  private static Bounds every(Type[] bounds) {
    // Every bound must match. On any false, result is false.
    return new Bounds(bounds, false);
  }

  private static Bounds any(Type[] bounds) {
    // Any bound matches. On any true, result is true.
    return new Bounds(bounds, true);
  }

  private static final class Bounds {
    private final Type[] bounds;
    private final boolean target;

    Bounds(Type[] bounds, boolean target) {
      this.bounds = bounds;
