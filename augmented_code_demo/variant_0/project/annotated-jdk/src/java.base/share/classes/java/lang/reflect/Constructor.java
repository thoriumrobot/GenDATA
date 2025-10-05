/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.lang.reflect;

    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.common.reflection.qual.NewInstance;
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
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import jdk.internal.reflect.ConstructorAccessor;
    @Positive
import jdk.internal.reflect.Reflection;
    @Positive
import jdk.internal.vm.annotation.ForceInline;
    @Positive
import sun.reflect.annotation.TypeAnnotation;
    @Positive
import sun.reflect.annotation.TypeAnnotationParser;
    @Positive
import sun.reflect.generics.repository.ConstructorRepository;
    @Positive
import sun.reflect.generics.factory.CoreReflectionFactory;
    @Positive
import sun.reflect.generics.factory.GenericsFactory;
    @Positive
import sun.reflect.generics.scope.ConstructorScope;
    @Positive
import java.lang.annotation.Annotation;
    @Positive
import java.lang.annotation.AnnotationFormatError;
    @Positive
import java.util.StringJoiner;

    @Positive
@CFComment({ "nullness: The type argument to Constructor is meaningless.", "Constructor<@NonNull String> and Constructor<@Nullable String> have the same", "meaning, but are unrelated by the Java type hierarchy.", "@Covariant makes Constructor<@NonNull String> a subtype of Constructor<@Nullable String>." })
    @Positive
@AnnotatedFor({ "lock", "nullness" })
    @Positive
@Covariant({ 0 })
    @Positive
public final class Constructor<T> extends Executable {

    @Positive
    @Override
    @Positive
    ConstructorRepository getGenericInfo();

    @Positive
    @Override
    @Positive
    Constructor<T> getRoot();

    @Positive
    Constructor<T> copy();

    @Positive
    @Override
    @Positive
    @CallerSensitive
    @Positive
    public void setAccessible(boolean flag);

    @Positive
    @Override
    @Positive
    void checkCanSetAccessible(Class<?> caller);

    @Positive
    @Override
    @Positive
    boolean hasGenericInformation();

    @Positive
    @Override
    @Positive
    byte[] getAnnotationBytes();

    @Positive
    @Override
    @Positive
    public Class<T> getDeclaringClass();

    @Positive
    @Override
    @Positive
    public String getName();

    @Positive
    @Override
    @Positive
    public int getModifiers();

    @Positive
    @Override
    @Positive
    @SuppressWarnings({ "rawtypes", "unchecked" })
    @Positive
    public TypeVariable<Constructor<T>>[] getTypeParameters();

    @Positive
    @Override
    @Positive
    Class<?>[] getSharedParameterTypes();

    @Positive
    @Override
    @Positive
    Class<?>[] getSharedExceptionTypes();

    @Positive
    @Override
    @Positive
    public Class<?>[] getParameterTypes();

    @Positive
    public int getParameterCount();

    @Positive
    @Override
    @Positive
    public Type[] getGenericParameterTypes();

    @Positive
    @Override
    @Positive
    public Class<?>[] getExceptionTypes();

    @Positive
    @Override
    @Positive
    public Type[] getGenericExceptionTypes();

    @Positive
    @Pure
    @Positive
    public boolean equals(@GuardSatisfied Constructor<T> this, @GuardSatisfied @Nullable Object obj);

    @Positive
    @Pure
    @Positive
    public int hashCode(@GuardSatisfied Constructor<T> this);

    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied Constructor<T> this);

    @Positive
    @Override
    @Positive
    void specificToStringHeader(StringBuilder sb);

    @Positive
    @Override
    @Positive
    String toShortString();

    @Positive
    @Override
    @Positive
    public String toGenericString();

    @Positive
    @Override
    @Positive
    void specificToGenericStringHeader(StringBuilder sb);

    @Positive
    @NewInstance
    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    @NonNull
    @Positive
    public T newInstance(Object... initargs) throws InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException;

    @Positive
    T newInstanceWithCaller(Object[] args, boolean checkAccess, Class<?> caller) throws InstantiationException, IllegalAccessException, InvocationTargetException;

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public boolean isVarArgs(@GuardSatisfied Constructor<T> this);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public boolean isSynthetic(@GuardSatisfied Constructor<T> this);

    @Positive
    ConstructorAccessor getConstructorAccessor();

    @Positive
    void setConstructorAccessor(ConstructorAccessor accessor);

    @Positive
    int getSlot();

    @Positive
    String getSignature();

    @Positive
    byte[] getRawAnnotations();

    @Positive
    byte[] getRawParameterAnnotations();

    @Positive
    @Override
    @Positive
    @Nullable
    @Positive
    public <T extends Annotation> T getAnnotation(Class<T> annotationClass);

    @Positive
    @Override
    @Positive
    public Annotation[] getDeclaredAnnotations();

    @Positive
    @Override
    @Positive
    public Annotation[][] getParameterAnnotations();

    @Positive
    @Override
    @Positive
    boolean handleParameterNumberMismatch(int resultLength, Class<?>[] parameterTypes);

    @Positive
    @Override
    @Positive
    public AnnotatedType getAnnotatedReturnType();

    @Positive
    @Override
    @Positive
    @Nullable
    @Positive
    public AnnotatedType getAnnotatedReceiverType();
    @Positive
}

// CFWR semantic augmentation - variant 0
