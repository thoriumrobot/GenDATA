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

import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.common.reflection.qual.Invoke;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;
import jdk.internal.access.SharedSecrets;
import jdk.internal.reflect.CallerSensitive;
import jdk.internal.reflect.MethodAccessor;
import jdk.internal.reflect.Reflection;
import jdk.internal.vm.annotation.ForceInline;
import jdk.internal.vm.annotation.IntrinsicCandidate;
import jdk.internal.vm.annotation.Stable;
import sun.reflect.annotation.ExceptionProxy;
import sun.reflect.annotation.TypeNotPresentExceptionProxy;
import sun.reflect.generics.repository.MethodRepository;
import sun.reflect.generics.factory.CoreReflectionFactory;
import sun.reflect.generics.factory.GenericsFactory;
import sun.reflect.generics.scope.MethodScope;
import sun.reflect.annotation.AnnotationType;
import sun.reflect.annotation.AnnotationParser;
import java.lang.annotation.Annotation;
import java.lang.annotation.AnnotationFormatError;
import java.nio.ByteBuffer;
import java.util.StringJoiner;

@AnnotatedFor({ "interning", "lock", "nullness" })
@SuppressWarnings({ "rawtypes" })
public final class Method extends Executable {

    @Override
    MethodRepository getGenericInfo();

    Method copy();

    Method leafCopy();

    @Override
    @CallerSensitive
    public void setAccessible(boolean flag);

    @Override
    void checkCanSetAccessible(Class<?> caller);

    @Override
    Method getRoot();

    @Override
    boolean hasGenericInformation();

    @Override
    byte[] getAnnotationBytes();

    @Override
    public Class<?> getDeclaringClass();

    @Override
    @Interned
    public String getName();

    @Override
    public int getModifiers();

    @Override
    @SuppressWarnings({ "rawtypes", "unchecked" })
    public TypeVariable<Method>[] getTypeParameters();

    @CFComment("lock/nullness: never returns null; returns Void instead")
    public Class<?> getReturnType();

    @CFComment("lock/nullness: never returns null; returns Void instead")
    public Type getGenericReturnType();

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
    public boolean equals(@GuardSatisfied Method this, @GuardSatisfied @Nullable Object obj);

    @Pure
    public int hashCode(@GuardSatisfied Method this);

    @SideEffectFree
    public String toString(@GuardSatisfied Method this);

    @Override
    void specificToStringHeader(StringBuilder sb);

    @Override
    String toShortString();

    String toShortSignature();

    @Override
    public String toGenericString();

    @Override
    void specificToGenericStringHeader(StringBuilder sb);

    @CFComment({ "lock/nullness: The method being invoked might be one that requires non-null", "arguments, or might be one that permits null.  We don't know which.", "Therefore, the Nullness Checker should conservatively issue a", "warning whenever null is passed, in order to give a guarantee that", "no nullness-related exception will be thrown by the invoked method." })
    @Invoke
    @CallerSensitive
    @ForceInline
    @IntrinsicCandidate
    @Nullable
    public Object invoke(Object obj, Object... args) throws IllegalAccessException, IllegalArgumentException, InvocationTargetException;

    @Pure
    public boolean isBridge(@GuardSatisfied Method this);

    @Pure
    @Override
    public boolean isVarArgs(@GuardSatisfied Method this);

    @Pure
    @Override
    public boolean isSynthetic(@GuardSatisfied Method this);

    public boolean isDefault();

    MethodAccessor getMethodAccessor();

    void setMethodAccessor(MethodAccessor accessor);

    @Nullable
    public Object getDefaultValue();

    @Override
    @Nullable
    public <T extends Annotation> T getAnnotation(Class<T> annotationClass);

    @Override
    public Annotation[] getDeclaredAnnotations();

    @Override
    public Annotation[][] getParameterAnnotations();

    @Override
    public AnnotatedType getAnnotatedReturnType();

    @Override
    boolean handleParameterNumberMismatch(int resultLength, Class<?>[] parameterTypes);
}
