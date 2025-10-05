// Source-based slice around line 1135
// Method: <com.google.common.reflect.TypeToken: boolean isOwnedBySubtypeOf(Type)>

        builder.add(Types.getArrayClass(of(t.getGenericComponentType()).getRawType()));
      }
    }.visit(runtimeType);
    // Cast from ImmutableSet<Class<?>> to ImmutableSet<Class<? super T>>
    @SuppressWarnings({"unchecked", "rawtypes"})
    ImmutableSet<Class<? super T>> result = (ImmutableSet) builder.build();
    return result;
  }

  private boolean isOwnedBySubtypeOf(Type supertype) {
    for (TypeToken<?> type : getTypes()) {
      Type ownerType = type.getOwnerTypeIfPresent();
      if (ownerType != null && of(ownerType).isSubtypeOf(supertype)) {
        return true;
      }
    }
    return false;
  }

  /**
