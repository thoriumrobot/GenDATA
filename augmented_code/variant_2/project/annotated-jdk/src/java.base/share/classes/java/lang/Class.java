/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1994, 2021, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.lang;

    @Positive
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.nullness.qual.PolyNull;
    @Positive
import org.checkerframework.checker.nullness.qual.UnknownKeyFor;
    @Positive
import org.checkerframework.checker.signature.qual.CanonicalName;
    @Positive
import org.checkerframework.checker.signature.qual.ClassGetName;
    @Positive
import org.checkerframework.checker.signature.qual.ClassGetSimpleName;
    @Positive
import org.checkerframework.checker.signature.qual.DotSeparatedIdentifiers;
    @Positive
import org.checkerframework.checker.signedness.qual.Signed;
    @Positive
import org.checkerframework.common.reflection.qual.ForName;
    @Positive
import org.checkerframework.common.reflection.qual.GetConstructor;
    @Positive
import org.checkerframework.common.reflection.qual.GetMethod;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import org.checkerframework.framework.qual.Covariant;
    @Positive
import java.lang.annotation.Annotation;
    @Positive
import java.lang.constant.ClassDesc;
    @Positive
import java.lang.invoke.TypeDescriptor;
    @Positive
import java.lang.invoke.MethodHandles;
    @Positive
import java.lang.module.ModuleReader;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.ObjectStreamField;
    @Positive
import java.lang.reflect.AnnotatedElement;
    @Positive
import java.lang.reflect.AnnotatedType;
    @Positive
import java.lang.reflect.Array;
    @Positive
import java.lang.reflect.Constructor;
    @Positive
import java.lang.reflect.Executable;
    @Positive
import java.lang.reflect.Field;
    @Positive
import java.lang.reflect.GenericArrayType;
    @Positive
import java.lang.reflect.GenericDeclaration;
    @Positive
import java.lang.reflect.InvocationTargetException;
    @Positive
import java.lang.reflect.Member;
    @Positive
import java.lang.reflect.Method;
    @Positive
import java.lang.reflect.Modifier;
    @Positive
import java.lang.reflect.Proxy;
    @Positive
import java.lang.reflect.RecordComponent;
    @Positive
import java.lang.reflect.Type;
    @Positive
import java.lang.reflect.TypeVariable;
    @Positive
import java.lang.constant.Constable;
    @Positive
import java.net.URL;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collection;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.LinkedHashMap;
    @Positive
import java.util.LinkedHashSet;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Optional;
    @Positive
import java.util.Set;
    @Positive
import java.util.stream.Collectors;
    @Positive
import jdk.internal.loader.BootLoader;
    @Positive
import jdk.internal.loader.BuiltinClassLoader;
    @Positive
import jdk.internal.misc.Unsafe;
    @Positive
import jdk.internal.module.Resources;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import jdk.internal.reflect.ConstantPool;
    @Positive
import jdk.internal.reflect.Reflection;
    @Positive
import jdk.internal.reflect.ReflectionFactory;
    @Positive
import jdk.internal.vm.annotation.ForceInline;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;
    @Positive
import sun.invoke.util.Wrapper;
    @Positive
import sun.reflect.generics.factory.CoreReflectionFactory;
    @Positive
import sun.reflect.generics.factory.GenericsFactory;
    @Positive
import sun.reflect.generics.repository.ClassRepository;
    @Positive
import sun.reflect.generics.repository.MethodRepository;
    @Positive
import sun.reflect.generics.repository.ConstructorRepository;
    @Positive
import sun.reflect.generics.scope.ClassScope;
    @Positive
import sun.security.util.SecurityConstants;
    @Positive
import sun.reflect.annotation.*;
    @Positive
import sun.reflect.misc.ReflectUtil;

    @Positive
@CFComment({ "interning: All instances of Class are interned.", "lock: public boolean isTypeAnnotationPresent(@GuardSatisfied Class<T> this,@GuardSatisfied Class<T><? extends java.lang.annotation.Annotation> annotationClass) { throw new RuntimeException(\"skeleton method\"); }", "public <M extends java.lang.annotation.Annotation> M getTypeAnnotation(Class<M> annotationClass) { throw new RuntimeException(\"skeleton method\"); }", "public java.lang.annotation.Annotation[] getTypeAnnotations() { throw new RuntimeException(\"skeleton method\"); }", "public java.lang.annotation.Annotation[] getDeclaredTypeAnnotations() { throw new RuntimeException(\"skeleton method\"); }", "nullness: The type argument to Class is meaningless.", "Class<@NonNull String> and Class<@Nullable String> have the same", "meaning, but are unrelated by the Java type hierarchy.", "@Covariant makes Class<@NonNull String> a subtype of Class<@Nullable String>." })
    @Positive
@AnnotatedFor({ "index", "interning", "lock", "nullness", "signature" })
    @Positive
@Covariant({ 0 })
    @Positive
@Interned
    @Positive
public final class Class<@UnknownKeyFor T> implements java.io.Serializable, GenericDeclaration, Type, AnnotatedElement, TypeDescriptor.OfField<Class<?>>, Constable {

    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied Class<T> this);

    @Positive
    public String toGenericString();

    @Positive
    static String typeVarBounds(TypeVariable<?> typeVar);

    @Positive
    @ForName
    @Positive
    @CallerSensitive
    @Positive
    public static Class<?> forName(@ClassGetName String className) throws ClassNotFoundException;

    @Positive
    @CallerSensitive
    @Positive
    public static Class<?> forName(@ClassGetName String name, boolean initialize, @Nullable ClassLoader loader) throws ClassNotFoundException;

    @Positive
    @SuppressWarnings("removal")
    @Positive
    @CallerSensitive
    @Positive
    public static Class<?> forName(Module module, String name);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    @CallerSensitive
    @Positive
    @Deprecated()
    @Positive
    @NonNull
    @Positive
    public T newInstance() throws InstantiationException, IllegalAccessException;

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = { "#1" }, result = true)
    @Positive
    @IntrinsicCandidate
    @Positive
    public native boolean isInstance(@GuardSatisfied Class<T> this, @Nullable Object obj);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public native boolean isAssignableFrom(@GuardSatisfied Class<T> this, Class<?> cls);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public native boolean isInterface(@GuardSatisfied Class<T> this);

    @Positive
    @EnsuresNonNullIf(expression = { "getComponentType()" }, result = true)
    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public native boolean isArray(@GuardSatisfied Class<T> this);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public native boolean isPrimitive(@GuardSatisfied Class<T> this);

    @Positive
    @Pure
    @Positive
    public boolean isAnnotation(@GuardSatisfied Class<T> this);

    @Positive
    @Pure
    @Positive
    public boolean isSynthetic(@GuardSatisfied Class<T> this);

    @Positive
    @CFComment({ "interning: In the Oracle JDK, the result of getName is interned", "signature: For a non-array non-primitive type, returns @BinaryName" })
    @Positive
    @Pure
    @Positive
    @ClassGetName
    @Positive
    @Interned
    @Positive
    public String getName();

    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    @Nullable
    @Positive
    public ClassLoader getClassLoader();

    @Positive
    ClassLoader getClassLoader0();

    @Positive
    public Module getModule();

    @Positive
    Object getClassData();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public TypeVariable<Class<T>>[] getTypeParameters();

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    @Nullable
    @Positive
    public native Class<? super T> getSuperclass(@GuardSatisfied Class<T> this);

    @Positive
    @Nullable
    @Positive
    public Type getGenericSuperclass();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public Package getPackage(@GuardSatisfied Class<T> this);

    @Positive
    @DotSeparatedIdentifiers
    @Positive
    public String getPackageName();

    @Positive
    @SideEffectFree
    @Positive
    public Class<?>[] getInterfaces(@GuardSatisfied Class<T> this);

    @Positive
    public Type[] getGenericInterfaces();

    @Positive
    @Pure
    @Positive
    @Nullable
    @Positive
    public Class<?> getComponentType(@GuardSatisfied Class<T> this);

    @Positive
    @Pure
    @Positive
    @IntrinsicCandidate
    @Positive
    public native int getModifiers(@GuardSatisfied Class<T> this);

    @Positive
    public native Object @Nullable [] getSigners();

    @Positive
    native void setSigners(Object[] signers);

    @Positive
    @CallerSensitive
    @Positive
    @Nullable
    @Positive
    public Method getEnclosingMethod() throws SecurityException;

    @Positive
    private static final class EnclosingMethodInfo {

    @Positive
        static void validate(Object[] enclosingInfo);

    @Positive
        boolean isPartial();

    @Positive
        boolean isConstructor();

    @Positive
        boolean isMethod();

    @Positive
        Class<?> getEnclosingClass();

    @Positive
        String getName();

    @Positive
        String getDescriptor();
    @Positive
    }

    @Positive
    @CallerSensitive
    @Positive
    @Nullable
    @Positive
    public Constructor<?> getEnclosingConstructor() throws SecurityException;

    @Positive
    @CallerSensitive
    @Positive
    @Nullable
    @Positive
    public Class<?> getDeclaringClass() throws SecurityException;

    @Positive
    @Pure
    @Positive
    @CallerSensitive
    @Positive
    @Nullable
    @Positive
    public Class<?> getEnclosingClass() throws SecurityException;

    @Positive
    @ClassGetSimpleName
    @Positive
    public String getSimpleName();

    @Positive
    public String getTypeName();

    @Positive
    @Nullable
    @Positive
    @CanonicalName
    @Positive
    public String getCanonicalName();

    @Positive
    @Pure
    @Positive
    public boolean isAnonymousClass(@GuardSatisfied Class<T> this);

    @Positive
    @Pure
    @Positive
    public boolean isLocalClass(@GuardSatisfied Class<T> this);

    @Positive
    @Pure
    @Positive
    public boolean isMemberClass(@GuardSatisfied Class<T> this);

    @Positive
    @SuppressWarnings("removal")
    @Positive
    @CallerSensitive
    @Positive
    public Class<?>[] getClasses();

    @Positive
    @CallerSensitive
    @Positive
    public Field[] getFields() throws SecurityException;

    @Positive
    @CallerSensitive
    @Positive
    public Method[] getMethods() throws SecurityException;

    @Positive
    @CallerSensitive
    @Positive
    public Constructor<?>[] getConstructors() throws SecurityException;

    @Positive
    @CallerSensitive
    @Positive
    public Field getField(String name) throws NoSuchFieldException, SecurityException;

    @Positive
    @Pure
    @Positive
    @GetMethod
    @Positive
    @CallerSensitive
    @Positive
    public Method getMethod(String name, Class<?>@Nullable ... parameterTypes) throws NoSuchMethodException, SecurityException;

    @Positive
    @GetConstructor
    @Positive
    @Pure
    @Positive
    @CallerSensitive
    @Positive
    public Constructor<T> getConstructor(Class<?>... parameterTypes) throws NoSuchMethodException, SecurityException;

    @Positive
    @CallerSensitive
    @Positive
    public Class<?>[] getDeclaredClasses() throws SecurityException;

    @Positive
    @CallerSensitive
    @Positive
    public Field[] getDeclaredFields() throws SecurityException;

    @Positive
    @CallerSensitive
    @Positive
    public RecordComponent[] getRecordComponents();

    @Positive
    @CallerSensitive
    @Positive
    public Method[] getDeclaredMethods() throws SecurityException;

    @Positive
    @CallerSensitive
    @Positive
    public Constructor<?>[] getDeclaredConstructors() throws SecurityException;

    @Positive
    @CallerSensitive
    @Positive
    public Field getDeclaredField(String name) throws NoSuchFieldException, SecurityException;

    @Positive
    @GetMethod
    @Positive
    @CallerSensitive
    @Positive
    public Method getDeclaredMethod(String name, Class<?>... parameterTypes) throws NoSuchMethodException, SecurityException;

    @Positive
    List<Method> getDeclaredPublicMethods(String name, Class<?>... parameterTypes);

    @Positive
    @CallerSensitive
    @Positive
    public Constructor<T> getDeclaredConstructor(Class<?>... parameterTypes) throws NoSuchMethodException, SecurityException;

    @Positive
    @CallerSensitive
    @Positive
    @Nullable
    @Positive
    public InputStream getResourceAsStream(String name);

    @Positive
    @CallerSensitive
    @Positive
    @Nullable
    @Positive
    public URL getResource(String name);

    @Positive
    public java.security.ProtectionDomain getProtectionDomain();

    @Positive
    java.security.ProtectionDomain protectionDomain();

    @Positive
    static native Class<?> getPrimitiveClass(String name);

    @Positive
    private static class Atomic {

    @Positive
        static <T> boolean casReflectionData(Class<?> clazz, SoftReference<ReflectionData<T>> oldData, SoftReference<ReflectionData<T>> newData);

    @Positive
        static boolean casAnnotationType(Class<?> clazz, AnnotationType oldType, AnnotationType newType);

    @Positive
        static boolean casAnnotationData(Class<?> clazz, AnnotationData oldData, AnnotationData newData);
    @Positive
    }

    @Positive
    private static class ReflectionData<T> {
    @Positive
    }

    @Positive
    native byte[] getRawAnnotations();

    @Positive
    native byte[] getRawTypeAnnotations();

    @Positive
    static byte[] getExecutableTypeAnnotationBytes(Executable ex);

    @Positive
    native ConstantPool getConstantPool();

    @Positive
    public boolean desiredAssertionStatus();

    @Positive
    @Pure
    @Positive
    public boolean isEnum(@GuardSatisfied Class<T> this);

    @Positive
    @Pure
    @Positive
    public boolean isRecord();

    @Positive
    @NonNull
    @Positive
    public T @Nullable [] getEnumConstants();

    @Positive
    @SuppressWarnings("removal")
    @Positive
    T[] getEnumConstantsShared();

    @Positive
    Map<String, @NonNull T> enumConstantDirectory();

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @IntrinsicCandidate
    @Positive
    @PolyNull
    @Positive
    @Signed
    @Positive
    public T cast(@PolyNull Object obj);

    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    public <U> Class<? extends U> asSubclass(Class<U> clazz);

    @Positive
    @Override
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Nullable
    @Positive
    public <A extends Annotation> A getAnnotation(Class<A> annotationClass);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public boolean isAnnotationPresent(@GuardSatisfied Class<T> this, @GuardSatisfied Class<? extends Annotation> annotationClass);

    @Positive
    @Override
    @Positive
    public <A extends Annotation> A[] getAnnotationsByType(Class<A> annotationClass);

    @Positive
    @Override
    @Positive
    public Annotation[] getAnnotations();

    @Positive
    @Override
    @Positive
    @SuppressWarnings("unchecked")
    @Positive
    @Nullable
    @Positive
    public <A extends Annotation> A getDeclaredAnnotation(Class<A> annotationClass);

    @Positive
    @Override
    @Positive
    public <A extends Annotation> A[] getDeclaredAnnotationsByType(Class<A> annotationClass);

    @Positive
    @Override
    @Positive
    public Annotation[] getDeclaredAnnotations();

    @Positive
    private static class AnnotationData {
    @Positive
    }

    @Positive
    boolean casAnnotationType(AnnotationType oldType, AnnotationType newType);

    @Positive
    AnnotationType getAnnotationType();

    @Positive
    Map<Class<? extends Annotation>, Annotation> getDeclaredAnnotationMap();

    @Positive
    public AnnotatedType getAnnotatedSuperclass();

    @Positive
    public AnnotatedType[] getAnnotatedInterfaces();

    @Positive
    @CallerSensitive
    @Positive
    public Class<?> getNestHost();

    @Positive
    public boolean isNestmateOf(Class<?> c);

    @Positive
    @CallerSensitive
    @Positive
    public Class<?>[] getNestMembers();

    @Positive
    @Override
    @Positive
    @SideEffectFree
    @Positive
    public String descriptorString();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public Class<?> componentType();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public Class<?> arrayType();

    @Positive
    @Override
    @Positive
    public Optional<ClassDesc> describeConstable();

    @Positive
    @IntrinsicCandidate
    @Positive
    @Pure
    @Positive
    public native boolean isHidden();

    @Positive
    @CallerSensitive
    @Positive
    public Class<?>[] getPermittedSubclasses();

    @Positive
    @Pure
    @Positive
    public boolean isSealed();
    @Positive
}
