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
import org.checkerframework.checker.interning.qual.Interned;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.common.reflection.qual.Invoke;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import jdk.internal.reflect.MethodAccessor;
    @Positive
import jdk.internal.reflect.Reflection;
    @Positive
import jdk.internal.vm.annotation.ForceInline;
    @Positive
import jdk.internal.vm.annotation.IntrinsicCandidate;
    @Positive
import jdk.internal.vm.annotation.Stable;
    @Positive
import sun.reflect.annotation.ExceptionProxy;
    @Positive
import sun.reflect.annotation.TypeNotPresentExceptionProxy;
    @Positive
import sun.reflect.generics.repository.MethodRepository;
    @Positive
import sun.reflect.generics.factory.CoreReflectionFactory;
    @Positive
import sun.reflect.generics.factory.GenericsFactory;
    @Positive
import sun.reflect.generics.scope.MethodScope;
    @Positive
import sun.reflect.annotation.AnnotationType;
    @Positive
import sun.reflect.annotation.AnnotationParser;
    @Positive
import java.lang.annotation.Annotation;
    @Positive
import java.lang.annotation.AnnotationFormatError;
    @Positive
import java.nio.ByteBuffer;
    @Positive
import java.util.StringJoiner;

    @Positive
@AnnotatedFor({ "interning", "lock", "nullness" })
    @Positive
@SuppressWarnings({ "rawtypes" })
    @Positive
public final class Method extends Executable {

    @Positive
    @Override
    @Positive
    MethodRepository getGenericInfo();

    @Positive
    Method copy();

    @Positive
    Method leafCopy();

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
    Method getRoot();

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
    public Class<?> getDeclaringClass();

    @Positive
    @Override
    @Positive
    @Interned
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
    public TypeVariable<Method>[] getTypeParameters();

    @Positive
    @CFComment("lock/nullness: never returns null; returns Void instead")
    @Positive
    public Class<?> getReturnType();

    @Positive
    @CFComment("lock/nullness: never returns null; returns Void instead")
    @Positive
    public Type getGenericReturnType();

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
    public boolean equals(@GuardSatisfied Method this, @GuardSatisfied @Nullable Object obj);

    @Positive
    @Pure
    @Positive
    public int hashCode(@GuardSatisfied Method this);

    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied Method this);

    @Positive
    @Override
    @Positive
    void specificToStringHeader(StringBuilder sb);

    @Positive
    @Override
    @Positive
    String toShortString();

    @Positive
    String toShortSignature();

    @Positive
    @Override
    @Positive
    public String toGenericString();

    @Positive
    @Override
    @Positive
    void specificToGenericStringHeader(StringBuilder sb);

    @Positive
    @CFComment({ "lock/nullness: The method being invoked might be one that requires non-null", "arguments, or might be one that permits null.  We don't know which.", "Therefore, the Nullness Checker should conservatively issue a", "warning whenever null is passed, in order to give a guarantee that", "no nullness-related exception will be thrown by the invoked method." })
    @Positive
    @Invoke
    @Positive
    @CallerSensitive
    @Positive
    @ForceInline
    @Positive
    @IntrinsicCandidate
    @Positive
    @Nullable
    @Positive
    public Object invoke(Object obj, Object... args) throws IllegalAccessException, IllegalArgumentException, InvocationTargetException;

    @Positive
    @Pure
    @Positive
    public boolean isBridge(@GuardSatisfied Method this);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public boolean isVarArgs(@GuardSatisfied Method this);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public boolean isSynthetic(@GuardSatisfied Method this);

    @Positive
    public boolean isDefault();

    @Positive
    MethodAccessor getMethodAccessor();

    @Positive
    void setMethodAccessor(MethodAccessor accessor);

    @Positive
    @Nullable
    @Positive
    public Object getDefaultValue();

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
    public AnnotatedType getAnnotatedReturnType();

    @Positive
    @Override
    @Positive
    boolean handleParameterNumberMismatch(int resultLength, Class<?>[] parameterTypes);
    @Positive
}

// CFWR semantic augmentation - variant 1
