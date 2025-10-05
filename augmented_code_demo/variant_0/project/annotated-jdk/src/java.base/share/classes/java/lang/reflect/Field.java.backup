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

import org.checkerframework.checker.initialization.qual.UnknownInitialization;
import org.checkerframework.checker.interning.qual.Interned;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;
import jdk.internal.access.SharedSecrets;
import jdk.internal.reflect.CallerSensitive;
import jdk.internal.reflect.FieldAccessor;
import jdk.internal.reflect.Reflection;
import jdk.internal.vm.annotation.ForceInline;
import sun.reflect.generics.repository.FieldRepository;
import sun.reflect.generics.factory.CoreReflectionFactory;
import sun.reflect.generics.factory.GenericsFactory;
import sun.reflect.generics.scope.ClassScope;
import java.lang.annotation.Annotation;
import java.util.Map;
import java.util.Objects;
import sun.reflect.annotation.AnnotationParser;
import sun.reflect.annotation.AnnotationSupport;
import sun.reflect.annotation.TypeAnnotation;
import sun.reflect.annotation.TypeAnnotationParser;

@CFComment({ "In general, the field value 'get' methods should take a top-qualified 'obj' parameter ", "and have a top-qualified return type; the field value 'set' methods should take a ", "top-qualified 'obj' parameter and a bottom-qualified 'value' parameter.", "nullness: the 'obj' parameter in 'get' or 'set' methods is @NonNull, because instance fields ", "require a receiver. Static field accesses need to suppress the errors.", "initialization: using fully-initialized types should make the typical use case easier.", "lock: require @GuardSatisfied to ensure type system soundness." })
@AnnotatedFor({ "interning", "lock", "nullness" })
public final class Field extends AccessibleObject implements Member {

    Field copy();

    @Override
    @CallerSensitive
    public void setAccessible(boolean flag);

    @Override
    void checkCanSetAccessible(Class<?> caller);

    @SideEffectFree
    @Override
    public Class<?> getDeclaringClass(@GuardSatisfied Field this);

    @SideEffectFree
    @Interned
    public String getName(@GuardSatisfied Field this);

    @Pure
    public int getModifiers(@GuardSatisfied Field this);

    @Pure
    public boolean isEnumConstant(@GuardSatisfied Field this);

    @Pure
    public boolean isSynthetic(@GuardSatisfied Field this);

    @SideEffectFree
    public Class<?> getType(@GuardSatisfied Field this);

    @SideEffectFree
    public Type getGenericType(@GuardSatisfied Field this);

    @Pure
    public boolean equals(@GuardSatisfied Field this, @GuardSatisfied @Nullable Object obj);

    @Pure
    public int hashCode(@GuardSatisfied Field this);

    @SideEffectFree
    public String toString(@GuardSatisfied Field this);

    @Override
    String toShortString();

    @SideEffectFree
    public String toGenericString(@GuardSatisfied Field this);

    @SideEffectFree
    @CallerSensitive
    @ForceInline
    @Nullable
    public Object get(@GuardSatisfied Field this, @GuardSatisfied Object obj) throws IllegalArgumentException, IllegalAccessException;

    @Pure
    @CallerSensitive
    @ForceInline
    public boolean getBoolean(@GuardSatisfied Field this, @GuardSatisfied Object obj) throws IllegalArgumentException, IllegalAccessException;

    @Pure
    @CallerSensitive
    @ForceInline
    public byte getByte(@GuardSatisfied Field this, @GuardSatisfied Object obj) throws IllegalArgumentException, IllegalAccessException;

    @Pure
    @CallerSensitive
    @ForceInline
    public char getChar(@GuardSatisfied Field this, @GuardSatisfied Object obj) throws IllegalArgumentException, IllegalAccessException;

    @Pure
    @CallerSensitive
    @ForceInline
    public short getShort(@GuardSatisfied Field this, @GuardSatisfied Object obj) throws IllegalArgumentException, IllegalAccessException;

    @Pure
    @CallerSensitive
    @ForceInline
    public int getInt(@GuardSatisfied Field this, @GuardSatisfied Object obj) throws IllegalArgumentException, IllegalAccessException;

    @Pure
    @CallerSensitive
    @ForceInline
    public long getLong(@GuardSatisfied Field this, @GuardSatisfied Object obj) throws IllegalArgumentException, IllegalAccessException;

    @Pure
    @CallerSensitive
    @ForceInline
    public float getFloat(@GuardSatisfied Field this, @GuardSatisfied Object obj) throws IllegalArgumentException, IllegalAccessException;

    @Pure
    @CallerSensitive
    @ForceInline
    public double getDouble(@GuardSatisfied Field this, @GuardSatisfied Object obj) throws IllegalArgumentException, IllegalAccessException;

    @CallerSensitive
    @ForceInline
    public void set(@GuardSatisfied Field this, @GuardSatisfied @UnknownInitialization Object obj, @GuardSatisfied @Interned Object value) throws IllegalArgumentException, IllegalAccessException;

    @CallerSensitive
    @ForceInline
    public void setBoolean(@GuardSatisfied Field this, @GuardSatisfied @UnknownInitialization Object obj, boolean z) throws IllegalArgumentException, IllegalAccessException;

    @CallerSensitive
    @ForceInline
    public void setByte(@GuardSatisfied Field this, @GuardSatisfied @UnknownInitialization Object obj, byte b) throws IllegalArgumentException, IllegalAccessException;

    @CallerSensitive
    @ForceInline
    public void setChar(@GuardSatisfied Field this, @GuardSatisfied @UnknownInitialization Object obj, char c) throws IllegalArgumentException, IllegalAccessException;

    @CallerSensitive
    @ForceInline
    public void setShort(@GuardSatisfied Field this, @GuardSatisfied @UnknownInitialization Object obj, short s) throws IllegalArgumentException, IllegalAccessException;

    @CallerSensitive
    @ForceInline
    public void setInt(@GuardSatisfied Field this, @GuardSatisfied @UnknownInitialization Object obj, int i) throws IllegalArgumentException, IllegalAccessException;

    @CallerSensitive
    @ForceInline
    public void setLong(@GuardSatisfied Field this, @GuardSatisfied @UnknownInitialization Object obj, long l) throws IllegalArgumentException, IllegalAccessException;

    @CallerSensitive
    @ForceInline
    public void setFloat(@GuardSatisfied Field this, @GuardSatisfied @UnknownInitialization Object obj, float f) throws IllegalArgumentException, IllegalAccessException;

    @CallerSensitive
    @ForceInline
    public void setDouble(@GuardSatisfied Field this, @GuardSatisfied @UnknownInitialization Object obj, double d) throws IllegalArgumentException, IllegalAccessException;

    @Override
    Field getRoot();

    boolean isTrustedFinal();

    @SideEffectFree
    @Override
    @Nullable
    public <T extends Annotation> T getAnnotation(@GuardSatisfied Field this, Class<T> annotationClass);

    @Override
    public <T extends Annotation> T[] getAnnotationsByType(Class<T> annotationClass);

    @SideEffectFree
    @Override
    public Annotation[] getDeclaredAnnotations(@GuardSatisfied Field this);

    public AnnotatedType getAnnotatedType();
}
