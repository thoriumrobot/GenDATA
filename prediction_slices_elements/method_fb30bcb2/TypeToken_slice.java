// Source-based slice around line 585
// Method: <com.google.common.reflect.TypeToken: TypeToken getComponentType()>

      return of(Primitives.unwrap(type));
    }
    return this;
  }

  /**
   * Returns the array component type if this type represents an array ({@code int[]}, {@code T[]},
   * {@code <? extends Map<String, Integer>[]>} etc.), or else {@code null} is returned.
   */
  public final @Nullable TypeToken<?> getComponentType() {
    Type componentType = Types.getComponentType(runtimeType);
    if (componentType == null) {
      return null;
    }
    return of(componentType);
  }

  /**
   * Returns the {@link Invokable} for {@code method}, which must be a member of {@code T}.
   *
