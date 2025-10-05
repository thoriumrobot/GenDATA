// Source-based slice around line 95
// Method: <com.google.common.reflect.TypeVisitor: void visitClass(Class)>

        succeeded = true;
      } finally {
        if (!succeeded) { // When the visitation failed, we don't want to ignore the second.
          visited.remove(type);
        }
      }
    }
  }

  void visitClass(Class<?> t) {}

  void visitGenericArrayType(GenericArrayType t) {}

  void visitParameterizedType(ParameterizedType t) {}

  void visitTypeVariable(TypeVariable<?> t) {}

  void visitWildcardType(WildcardType t) {}
}
