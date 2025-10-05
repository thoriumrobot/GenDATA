// Source-based slice around line 1149
// Method: <com.google.common.reflect.TypeToken: Type getOwnerTypeIfPresent()>

      }
    }
    return false;
  }

  /**
   * Returns the owner type of a {@link ParameterizedType} or enclosing class of a {@link Class}, or
   * null otherwise.
   */
  private @Nullable Type getOwnerTypeIfPresent() {
    if (runtimeType instanceof ParameterizedType) {
      return ((ParameterizedType) runtimeType).getOwnerType();
    } else if (runtimeType instanceof Class<?>) {
      return ((Class<?>) runtimeType).getEnclosingClass();
    } else {
      return null;
    }
  }

  /**
