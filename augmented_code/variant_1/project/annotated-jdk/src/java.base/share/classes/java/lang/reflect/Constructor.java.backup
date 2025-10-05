/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.lang.reflect;

import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.common.reflection.qual.NewInstance;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;
import org.checkerframework.framework.qual.Covariant;
import jdk.internal.access.SharedSecrets;
import jdk.internal.reflect.CallerSensitive;
import jdk.internal.reflect.ConstructorAccessor;
import jdk.internal.reflect.Reflection;
import jdk.internal.vm.annotation.ForceInline;
import sun.reflect.annotation.TypeAnnotation;
import sun.reflect.annotation.TypeAnnotationParser;
import sun.reflect.generics.repository.ConstructorRepository;
import sun.reflect.generics.factory.CoreReflectionFactory;
import sun.reflect.generics.factory.GenericsFactory;
import sun.reflect.generics.scope.ConstructorScope;
import java.lang.annotation.Annotation;
import java.lang.annotation.AnnotationFormatError;
import java.util.StringJoiner;

@CFComment({ "nullness: The type argument to Constructor is meaningless.", "Constructor<@NonNull String> and Constructor<@Nullable String> have the same", "meaning, but are unrelated by the Java type hierarchy.", "@Covariant makes Constructor<@NonNull String> a subtype of Constructor<@Nullable String>." })
@AnnotatedFor({ "lock", "nullness" })
@Covariant({ 0 })
public final class Constructor<T> extends Executable {

    @Override
    ConstructorRepository getGenericInfo();

    @Override
    Constructor<T> getRoot();

    Constructor<T> copy();

    @Override
    @CallerSensitive
    public void setAccessible(boolean flag);

    @Override
    void checkCanSetAccessible(Class<?> caller);

    @Override
    boolean hasGenericInformation();

    @Override
    byte[] getAnnotationBytes();

    @Override
    public Class<T> getDeclaringClass();

    @Override
    public String getName();

    @Override
    public int getModifiers();

    @Override
    @SuppressWarnings({ "rawtypes", "unchecked" })
    public TypeVariable<Constructor<T>>[] getTypeParameters();

    @Override
    Class<?>[] getSharedParameterTypes();

    @Override
    Class<?>[] getSharedExceptionTypes();

    @Override
    public Class<?>[] getParameterTypes();

    public int getParameterCount();

    @Override
    public Type[] getGenericParameterTypes();

    @Override
    public Class<?>[] getExceptionTypes();

    @Override
    public Type[] getGenericExceptionTypes();

    @Pure
    public boolean equals(@GuardSatisfied Constructor<T> this, @GuardSatisfied @Nullable Object obj);

    @Pure
    public int hashCode(@GuardSatisfied Constructor<T> this);

    @SideEffectFree
    public String toString(@GuardSatisfied Constructor<T> this);

    @Override
    void specificToStringHeader(StringBuilder sb);

    @Override
    String toShortString();

    @Override
    public String toGenericString();

    @Override
    void specificToGenericStringHeader(StringBuilder sb);

    @NewInstance
    @CallerSensitive
    @ForceInline
    @NonNull
    public T newInstance(Object... initargs) throws InstantiationException, IllegalAccessException, IllegalArgumentException, InvocationTargetException;

    T newInstanceWithCaller(Object[] args, boolean checkAccess, Class<?> caller) throws InstantiationException, IllegalAccessException, InvocationTargetException;

    @Pure
    @Override
    public boolean isVarArgs(@GuardSatisfied Constructor<T> this);

    @Pure
    @Override
    public boolean isSynthetic(@GuardSatisfied Constructor<T> this);

    ConstructorAccessor getConstructorAccessor();

    void setConstructorAccessor(ConstructorAccessor accessor);

    int getSlot();

    String getSignature();

    byte[] getRawAnnotations();

    byte[] getRawParameterAnnotations();

    @Override
    @Nullable
    public <T extends Annotation> T getAnnotation(Class<T> annotationClass);

    @Override
    public Annotation[] getDeclaredAnnotations();

    @Override
    public Annotation[][] getParameterAnnotations();

    @Override
    boolean handleParameterNumberMismatch(int resultLength, Class<?>[] parameterTypes);

    @Override
    public AnnotatedType getAnnotatedReturnType();

    @Override
    @Nullable
    public AnnotatedType getAnnotatedReceiverType();
}
